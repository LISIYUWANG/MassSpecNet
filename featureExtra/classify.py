# -*- coding: utf-8 -*-
# @Time    : 2022/1/14 0:57
# @Author  : naptmn
# @File    : classify.py
# @Software: PyCharm
import pickle
from time import sleep
from sklearn .model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
def test(train_feature,train_label,test_feature,test_label):
    sf = StratifiedKFold(n_splits=5,shuffle=False)
    for train_index, test_index in sf.split(train_feature, train_label):
        X_train, X_test = train_feature[train_index], test_feature[test_index]
        y_train, y_test = train_label[train_index], test_label[test_index]
        svc = SVC(probability=True)
        svc.fit(X_train, y_train)
        prodict = svc.predict(X_test)
        # prob = svc.predict_proba(y_test)[:, 1]
        acc_test = accuracy_score(y_test, prodict)
        # auc_test = roc_auc_score(y_test, prob)
        print(acc_test)
def classification_svc(train_feature,train_label,test_feature,test_label):
    print('svc')
    PARAMETER_GRID = {
        'C': [1e-1, 1e0, 1e1, 1e2],
        'kernel': ['linear', 'poly', 'rbf'],
        #'n_estimators': [100, 500],
    }
    svc = SVC(probability=True)
    clf = GridSearchCV(svc, PARAMETER_GRID, cv=3, scoring='accuracy', verbose=3)
    clf.fit(train_feature, train_label)
    #svc.fit(train_feature, train_label)
    print(clf.best_params_)
    train_score = clf.score(train_feature, train_label)
    acc_train = train_score
    prob = clf.predict_proba(test_feature)[:, 1]
    prodict = clf.predict(test_feature)
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)  # 验证集上的auc值
    return acc_train, acc_test, auc_test, clf.best_params_
def classification_rf(train_feature,train_label,test_feature,test_label):
    print('rf')
    PARAMETER_GRID = {
        'n_estimators': [100, 500],
    }
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, PARAMETER_GRID, cv=3, scoring='roc_auc', verbose=3)
    clf.fit(train_feature, train_label)
    print(clf.best_params_)
    train_score = clf.score(train_feature, train_label)
    acc_train = train_score
    prob = clf.predict_proba(test_feature)[:, 1]
    prodict = clf.predict(test_feature)
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)  # 验证集上的auc值
    return acc_train, acc_test, auc_test, clf.best_params_
def classification_lr(train_feature,train_label,test_feature,test_label):
    print('lr')
    PARAMETER_GRID = {
        'C': [1e-1, 1e0, 1e1, 1e2],
        #'kernel': ['linear', 'poly', 'rbf'],
        #'n_estimators': [100, 500],
    }
    lr = LogisticRegression(max_iter=300)
    clf = GridSearchCV(lr, PARAMETER_GRID, cv=3, scoring='accuracy', verbose=3)
    clf.fit(train_feature, train_label)
    print(clf.best_params_)
    #lr.fit(train_feature, train_label)
    train_score = clf.score(train_feature, train_label)
    test_score = clf.score(test_feature, test_label)
    acc_train = train_score
    prob = clf.predict_proba(test_feature)[:, 1]
    prodict = clf.predict(test_feature)
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)  # 验证集上的auc值
    return acc_train, acc_test, auc_test, clf.best_params_
def classification_xgboost(train_feature,train_label,test_feature,test_label):
    print('xgb')
    PARAMETER_GRID = {
        'n_estimators': [100, 500],
    }
    xgboost = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False)
    clf = GridSearchCV(estimator=xgboost, param_grid=PARAMETER_GRID, cv=3, scoring='roc_auc', verbose=3)
    clf.fit(train_feature, train_label)
    print(clf.best_params_)
    train_score = clf.score(train_feature, train_label)
    auc_train = train_score
    prob = clf.predict_proba(test_feature)[:, 1]
    prodict = clf.predict(test_feature)
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)  # 验证集上的auc值
    return auc_train, acc_test, auc_test, clf.best_params_
def classification_multinomialnb(train_feature,train_label,test_feature,test_label):
    PARAMETER_GRID = {
        'alpha': [0.1, 0.5, 1.0],
    }
    bays = MultinomialNB()
    clf = GridSearchCV(bays, PARAMETER_GRID, cv=3, scoring='roc_auc')
    clf.fit(train_feature, train_label)
    print(clf.best_params_)
    train_score = clf.score(train_feature, train_label)
    acc_train = train_score
    prob = clf.predict_proba(test_feature)[:, 1]
    prodict = clf.predict(test_feature)
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)  # 验证集上的auc值
    return acc_train, acc_test, auc_test, clf.best_params_
def classification_gaussiannb(train_feature,train_label,test_feature,test_label):
    print('gaussiannb')
    bays = GaussianNB()
    bays.fit(train_feature, train_label)
    prodict = bays.predict(test_feature)
    prob = bays.predict_proba(test_feature)[:, 1]
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)
    print(acc_test, auc_test)
    return acc_test, auc_test

def write(classifyName,fcName ,bestParams, acc, auc):
    output.write(classifyName)
    output.write(',')
    output.write(fcName)
    output.write(',')
    output.write(str(bestParams))
    output.write(',')
    output.write(str(acc))
    output.write(',')
    output.write(str(auc))
    output.write('\n')
if __name__ =='__main__':
    NAME = 'PXD007088'#'PXD008383'#
    output = open('result_fc'+NAME+'.csv', 'w')
    output.write('classifyName')
    output.write(',')
    output.write('fcName')
    output.write(',')
    output.write('bestParams')
    output.write(',')
    output.write('acc')
    output.write(',')
    output.write('auc')
    output.write('\n')
    train_path = ['./features/feature_train_fc1_before'+NAME+'_trans.pkl'
                  ,'./features/feature_train_fc2_before'+NAME+'_trans.pkl'
                  , './features/feature_train_fc3_before'+NAME+'_trans.pkl'
               ]
    test_path = ['./features/feature_test_fc1_before'+NAME+'_trans.pkl'
                  ,'./features/feature_test_fc2_before'+NAME+'_trans.pkl'
                  , './features/feature_test_fc3_before'+NAME+'_trans.pkl'
                 ]
    train_label_path = './features/labels_train'+NAME+'n.pkl'
    test_label_path = './features/labels_test'+NAME+'n.pkl'
    # train_path = ['train_feature_fc2_af_data7_3_orgin_rui7.pkl']
    # test_path = ['test_feature_fc2_af_data7_3_orgin_rui7.pkl']
    # train_label_path = 'train_label_fc2_af_data7_3_orgin_rui7.pkl'
    # test_label_path = 'test_label_fc2_af_data7_3_orgin_rui7.pkl'
    name = ['fc1_before', 'fc2_before', 'fc3_before']  #, 'fc3_after'
    # 寻优参数范围
    # param_grid_lr = [
    #     {
    #         'C':[0.1,1,10,100]
    #     }]
    # param_grid_svc = [
    #     {
    #         'C': [0.1, 1, 10, 100],
    #         'kernel': ['linear', 'poly', 'rbf']
    #     }]
    # param_grid_rf = [
    #     {
    #         'n_estimators':[100, 500]
    #     }]
    # param_grid_xgb = [
    #     {
    #         'n_estimators': [100, 500]
    #     }]
    i=0
    for path in train_path:
        features_train = pickle.load(open(path, 'rb'))
        labels_train = pickle.load(open(train_label_path, 'rb'))
        features_test = pickle.load(open(test_path[i], 'rb'))
        labels_test = pickle.load(open(test_label_path, 'rb'))
        features_train = np.array(features_train)
        labels_train = np.array(labels_train)
        features_test = np.array(features_test)
        labels_test = np.array(labels_test)
        # print(features_train)
        # print(labels_train)
        # test(train_feature=features_train,train_label=labels_train,test_feature=features_test,
        #                         test_label=labels_test)
        _, acc, auc, params = classification_lr(train_feature=features_train,train_label=labels_train,test_feature=features_test,
                                test_label=labels_test)
        write('lr', name[i], params, acc,auc)
        _, acc, auc, params = classification_svc(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                                test_label=labels_test)
        write('svc', name[i], params, acc, auc)
        _, acc, auc, params = classification_rf(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                                test_label=labels_test)
        write('rf', name[i], params, acc, auc)
        _, acc, auc, params = classification_xgboost(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                                test_label=labels_test)
        write('xgboost', name[i], params, acc, auc)
        # print(classification_multinomialnb(train_feature=features_train, train_label=labels_train, test_feature=features_test,
        #                         test_label=labels_test))
        acc, auc = classification_gaussiannb(train_feature=features_train, train_label=labels_train,
                           test_feature=features_test,
                           test_label=labels_test)
        write('gaussiannb', name[i], params,acc, auc)
        i = i+1

        #
        # print(np.array(features_train).shape)
        # grid_lr = GridSearchCV(LogisticRegression(),
        #                        param_grid_lr,
        #                        cv=6,
        #                        scoring='roc_auc',
        #                        verbose=0)
        # grid_lr.fit(features_train, labels_train)
        # print(grid_lr.best_params_)
        # print(grid_lr.cv_results_)
        # best_lr = grid_lr.scoring
        # print(best_lr)
        # #print(best_lr)
        # # pip_lr  = Pipeline([
        # #     ('VarianceThreshold',VarianceThreshold()) # 默认方差为0 去除常量特征
        # #     ('MinMax',MinMaxScaler) # 归一化到0-1 默认为0-1
        # # ])
