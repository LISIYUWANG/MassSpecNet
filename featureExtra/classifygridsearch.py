# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 17:11
# @Author  : naptmn
# @File    : classifygridsearch.py
# @Software: PyCharm
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
def classification_svc(train_feature,train_label,test_feature,test_label,layername):
    print('svc')
    # 筛选范围
    dic = {'C':[1e-1, 1e0, 1e1, 1e2],
           'kernel': ('linear', 'poly', 'rbf')
           }
    best = None
    bestparams = None
    auc = 0
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_feature, train_label, test_size=0.3, stratify=train_label, random_state=0)
    for c in dic['C']:
        for ker in dic['kernel']:
            svc = SVC(C=c, kernel=ker, probability=True)
            svc.fit(Xtrain,Ytrain)
            # 打分
            prodict = svc.predict(Xtest)
            prob = svc.predict_proba(Xtest)[:, 1]
            acc_test = accuracy_score(Ytest, prodict)
            auc_test = roc_auc_score(Ytest, prob)
            tempparams = 'C='+str(c)+'; kernel='+ker
            writeall('svc',layername, tempparams, acc_test, auc_test)
            if auc_test > auc:
                auc = auc_test
                best = svc
                bestparams = tempparams
            print('C',c,'kernel',ker,'score',acc_test, auc_test)
    prodict = best.predict(test_feature)
    prob = best.predict_proba(test_feature)[:, 1]
    acc_best = accuracy_score(test_label, prodict)
    auc_best = roc_auc_score(test_label, prob)
    # 保存最好模型的test效果
    print('best',acc_best,auc_best)
    write('svc', layername, bestparams, acc_best, auc_best)
    # return acc_best,auc_best,bestparams
def classification_rf(train_feature,train_label,test_feature,test_label,layername):
    print('rf')
    # 筛选范围
    dic = {'n_estimators':[100, 500]
           }
    best = None
    bestparams = None
    auc = 0
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_feature, train_label, test_size=0.3, stratify=train_label, random_state=0)
    for n_estimators in dic['n_estimators']:
            rf = RandomForestClassifier(n_estimators=n_estimators)
            rf.fit(Xtrain,Ytrain)
            # 打分
            prodict = rf.predict(Xtest)
            prob = rf.predict_proba(Xtest)[:, 1]
            acc_test = accuracy_score(Ytest, prodict)
            auc_test = roc_auc_score(Ytest, prob)
            tempparams = 'n_estimators='+str(n_estimators)
            writeall('rf',layername, tempparams, acc_test, auc_test)
            if auc_test > auc:
                auc = auc_test
                best = rf
                bestparams = tempparams
    prodict = best.predict(test_feature)
    prob = best.predict_proba(test_feature)[:, 1]
    acc_best = accuracy_score(test_label, prodict)
    auc_best = roc_auc_score(test_label, prob)
    write('rf', layername, bestparams, acc_best, auc_best)
def classification_lr(train_feature,train_label,test_feature,test_label,layername):
    print('lr')
    # 筛选范围
    dic = {'C':[1e-1, 1e0, 1e1, 1e2]
           }
    best = None
    bestparams = None
    auc = 0
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_feature, train_label, test_size=0.3, stratify=train_label, random_state=0)
    for c in dic['C']:
            lr = LogisticRegression(C=c)
            lr.fit(Xtrain,Ytrain)
            # 打分
            prodict = lr.predict(Xtest)
            prob = lr.predict_proba(Xtest)[:, 1]
            acc_test = accuracy_score(Ytest, prodict)
            auc_test = roc_auc_score(Ytest, prob)
            tempparams = 'c='+str(c)
            writeall('lr',layername, tempparams, acc_test, auc_test)
            if auc_test > auc:
                auc = auc_test
                best = lr
                bestparams = tempparams
    prodict = best.predict(test_feature)
    prob = best.predict_proba(test_feature)[:, 1]
    acc_best = accuracy_score(test_label, prodict)
    auc_best = roc_auc_score(test_label, prob)
    write('lr', layername, bestparams, acc_best, auc_best)
def classification_xgboost(train_feature,train_label,test_feature,test_label,layername):
    print('xgb')
    # 筛选范围
    dic = {'n_estimators':[100, 500]
           }
    best = None
    bestparams = None
    auc = 0
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_feature, train_label, test_size=0.3, stratify=train_label, random_state=0)
    for n_estimators in dic['n_estimators']:
            xgboost = XGBClassifier(eval_metric=['logloss','auc','error'],use_label_encoder=False)
            xgboost.fit(Xtrain,Ytrain)
            # 打分
            prodict = xgboost.predict(Xtest)
            prob = xgboost.predict_proba(Xtest)[:, 1]
            acc_test = accuracy_score(Ytest, prodict)
            auc_test = roc_auc_score(Ytest, prob)
            tempparams = 'n_estimators='+str(n_estimators)
            writeall('xgboost',layername, tempparams, acc_test, auc_test)
            if auc_test > auc:
                auc = auc_test
                best = xgboost
                bestparams = tempparams
    prodict = best.predict(test_feature)
    prob = best.predict_proba(test_feature)[:, 1]
    acc_best = accuracy_score(test_label, prodict)
    auc_best = roc_auc_score(test_label, prob)
    write('xgboost', layername, bestparams, acc_best, auc_best)
def classification_gaussiannb(train_feature,train_label,test_feature,test_label,layername):
    # 这个没参数
    print('gaussiannb')
    bays = GaussianNB()
    bays.fit(train_feature, train_label)
    prodict = bays.predict(test_feature)
    prob = bays.predict_proba(test_feature)[:, 1]
    acc_test = accuracy_score(test_label, prodict)
    auc_test = roc_auc_score(test_label, prob)
    print(acc_test, auc_test)
    bestparams = 'NoneParams'
    write('gaussiannb', layername, bestparams, acc_test, auc_test)
    # return acc_test, auc_test
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
def writeall(classifyName,fcName ,Params, acc, auc):
    outputall.write(classifyName)
    outputall.write(',')
    outputall.write(fcName)
    outputall.write(',')
    outputall.write(str(Params))
    outputall.write(',')
    outputall.write(str(acc))
    outputall.write(',')
    outputall.write(str(auc))
    outputall.write('\n')
if __name__ =='__main__':
    # 最好模型
    NAME = 'PXD008383' #'PXD007088' #
    output = open(NAME +'_result_fc.csv', 'w')
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
    # 中间过程
    outputall = open(NAME +'_result_fc_all.csv', 'w')
    outputall.write('classifyName')
    outputall.write(',')
    outputall.write('fcName')
    outputall.write(',')
    outputall.write('Params')
    outputall.write(',')
    outputall.write('acc')
    outputall.write(',')
    outputall.write('auc')
    outputall.write('\n')
    # #PXD007088
    # train_path = ['./features/feature_train_ConvBNActivationV2PXD007088_trans.pkl'
    #               ,'./features/feature_train_ConvBNActivationV3PXD007088_trans.pkl'
    #
    #               ,'./features/feature_train_dropoutBeforePXD007088_trans.pkl'
    #               ,'./features/feature_train_fc1_beforePXD007088_trans.pkl'
    #               ,'./features/feature_train_fc2_beforePXD007088_trans.pkl'
    #               ,'./features/feature_train_fc3_beforePXD007088_trans.pkl'
    #              ]
    # test_path = ['./features/feature_test_ConvBNActivationV2PXD007088_trans.pkl'
    #               ,'./features/feature_test_ConvBNActivationV3PXD007088_trans.pkl'
    #
    #               ,'./features/feature_test_dropoutBeforePXD007088_trans.pkl'
    #               ,'./features/feature_test_fc1_beforePXD007088_trans.pkl'
    #               ,'./features/feature_test_fc2_beforePXD007088_trans.pkl'
    #               ,'./features/feature_test_fc3_beforePXD007088_trans.pkl'
    #               ]
    # train_label_path = './features/labels_trainPXD007088n.pkl'
    # test_label_path = './features/labels_testPXD007088n.pkl'
    # # PXD008383
    train_path = ['./features/feature_train_ConvBNActivationV2PXD008383_trans.pkl'
                  ,'./features/feature_train_ConvBNActivationV3PXD008383_trans.pkl'

                  ,'./features/feature_train_dropoutBeforePXD008383_trans.pkl'
                  ,'./features/feature_train_fc1_beforePXD008383_trans.pkl'
                  ,'./features/feature_train_fc2_beforePXD008383_trans.pkl'
                  ,'./features/feature_train_fc3_beforePXD008383_trans.pkl'
                  ]
    test_path = ['./features/feature_test_ConvBNActivationV2PXD008383_trans.pkl'
                  ,'./features/feature_test_ConvBNActivationV3PXD008383_trans.pkl'

                  ,'./features/feature_test_dropoutBeforePXD008383_trans.pkl'
                  ,'./features/feature_test_fc1_beforePXD008383_trans.pkl'
                  ,'./features/feature_test_fc2_beforePXD008383_trans.pkl'
                  ,'./features/feature_test_fc3_beforePXD008383_trans.pkl'
                ]
    train_label_path = './features/labels_trainPXD008383n.pkl'
    test_label_path = './features/labels_testPXD008383n.pkl'
    #name = ['ConvBNActivationV2','ConvBNActivationV3','dropoutBefore','fc1_before', 'fc2_before', 'fc3_before']
    name = ['MobileNetV2','MobileNetV3_small','GhostNet','fc1_before', 'fc2_before', 'fc3_before']


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
        classification_svc(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                                test_label=labels_test,layername=name[i])
        classification_rf(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                           test_label=labels_test, layername=name[i])
        classification_lr(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                           test_label=labels_test, layername=name[i])
        classification_xgboost(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                           test_label=labels_test, layername=name[i])
        classification_gaussiannb(train_feature=features_train, train_label=labels_train, test_feature=features_test,
                           test_label=labels_test, layername=name[i])
        i = i+1

