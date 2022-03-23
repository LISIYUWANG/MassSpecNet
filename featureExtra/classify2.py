# -*- coding: utf-8 -*-
# @Time    : 2022/1/16 15:40
# @Author  : naptmn
# @File    : classify2.py
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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
def classification_svc(train_feature,train_label,test_feature,test_label):
    print('svc')
    PARAMETER_GRID = {
        'C': [1e-1, 1e0, 1e1, 1e2],
        'kernel': ['linear', 'poly', 'rbf']
    }
    svc = SVC(probability=True)
    svc.fit(train_feature, train_label)
    prodict = svc.predict(test_feature)
    prob = svc.predict_proba(test_feature)[:, 1]
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)
    print(acc_test, auc_test)
    return acc_test, auc_test
def classification_rf(train_feature,train_label,test_feature,test_label):
    print('rf')
    PARAMETER_GRID = {
        'n_estimators': [100, 500],
    }
    rf = RandomForestClassifier()
    rf.fit(train_feature, train_label)
    prodict = rf.predict(test_feature)
    prob = rf.predict_proba(test_feature)[:, 1]
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)
    print(acc_test, auc_test)
    return acc_test, auc_test
def classification_lr(train_feature,train_label,test_feature,test_label):
    print('lr')
    PARAMETER_GRID = {
        'C': [1e-1, 1e0, 1e1, 1e2]
    }
    lr = LogisticRegression()
    lr.fit(train_feature, train_label)
    prodict = lr.predict(test_feature)
    prob = lr.predict_proba(test_feature)[:, 1]
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)
    print(acc_test, auc_test)
    return acc_test, auc_test
def classification_xgboost(train_feature,train_label,test_feature,test_label):
    print('xgb')
    PARAMETER_GRID = {
        'n_estimators': [100, 500],
    }
    xgboost = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False)
    xgboost.fit(train_feature, train_label)
    prodict = xgboost.predict(test_feature)
    prob = xgboost.predict_proba(test_feature)[:, 1]
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)
    print(acc_test, auc_test)
    return acc_test, auc_test
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
def write(classifyName, fcName, acc, auc):
    output.write(classifyName)
    output.write(',')
    output.write(fcName)
    output.write(',')
    output.write(str(acc))
    output.write(',')
    output.write(str(auc))
    output.write('\n')
if __name__ =='__main__':
    NAME = 'PXD007088'#'PXD008383'#'data7_3_orgin'     #
    web_ls = ['ConvBNActivationV3','ConvBNActivationV2','dropoutBefore']
    for name2 in web_ls:
        output = open('result_'+name2+'_'+NAME+'_trans.csv', 'w')
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
        train_path = './features/feature_train_'+name2+NAME+'_trans.pkl'
        test_path = './features/feature_test_'+name2+NAME+'_trans.pkl'

        train_label_path = './features/labels_train'+NAME+'n.pkl'
        test_label_path = './features/labels_test'+NAME+'n.pkl'
        #name = ['fc1_before', 'fc2_before', 'fc3_before', 'fc3_after']
        i = 0
        features_train = pickle.load(open(train_path, 'rb'))
        labels_train = pickle.load(open(train_label_path, 'rb'))
        features_test = pickle.load(open(test_path, 'rb'))
        labels_test = pickle.load(open(test_label_path, 'rb'))
        features_train = np.array(features_train)
        labels_train = np.array(labels_train)
        features_test = np.array(features_test)
        labels_test = np.array(labels_test)
        acc, auc = classification_svc(train_feature=features_train, train_label=labels_train,
                               test_feature=features_test,
                               test_label=labels_test)
        write('svc', name2,acc, auc)
        acc, auc = classification_rf(train_feature=features_train, train_label=labels_train,
                               test_feature=features_test,
                               test_label=labels_test)
        write('rf', name2, acc, auc)
        acc, auc = classification_lr(train_feature=features_train, train_label=labels_train,
                           test_feature=features_test,
                           test_label=labels_test)
        write('lr', name2, acc, auc)
        acc, auc = classification_xgboost(train_feature=features_train, train_label=labels_train,
                           test_feature=features_test,
                           test_label=labels_test)
        write('xgboost', name2, acc, auc)
        acc, auc = classification_gaussiannb(train_feature=features_train, train_label=labels_train,
                           test_feature=features_test,
                           test_label=labels_test)
        write('gaussiannb', name2, acc, auc)
        i = i + 1