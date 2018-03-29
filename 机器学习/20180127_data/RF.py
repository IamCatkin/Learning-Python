#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/27 14:44
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : RF.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
#########     参数     ###########
path = '.\\data\\'
descriptors = ["cats","ecfp4","maccs","moe2d"]
cycles = 10

def getproteins():
    files = os.listdir(path)  
    names = set()
    for file in files:
        name = file[0:file.rfind('-', 1)]
        names.add(name)
    names = list(names)
    return names

def model(protein,descriptor,i):
    if i == 0:
        with open('.\\result\\'+protein+'-'+descriptor+'.csv','a+') as f:
            f.write('ACC'+','+'SPE'+','+'SEN'+','+'AUC'+'\n')
    data = pd.read_table(path+protein+'-'+descriptor+'.csv',delimiter=',')   
    X = data.iloc[:,3:len(data.columns)]
    y = data.iloc[:,2]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
    answer_rf = rf.predict(X_test)
    
    y_score = rf.predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
    confusion = metrics.confusion_matrix(y_test,answer_rf)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    auc = metrics.auc(fpr,tpr)
    acc = metrics.accuracy_score(y_test,answer_rf)
    sen = TP / (TP+FN)
    spe = TN / (TN+FP)
    print('========='+protein+'-'+descriptor+'========')
    print('The accuracy is: %.3f' %acc)
    print('The specificity is: %.3f' %spe)
    print('The sensitivity is: %.3f' %sen)
    print('The auc is: %.3f' %auc)
    print('============================')
    with open('.\\result\\'+protein+'-'+descriptor+'.csv','a+') as f:
        f.write('{:.3f}'.format(float(acc))+','+'{:.3f}'.format(float(spe))+','+'{:.3f}'.format(float(sen))+','+'{:.3f}'.format(float(auc))+'\n')

if __name__ == '__main__':
    proteins = getproteins()
    for protein in proteins:
        for descriptor in descriptors:
            for i in range(cycles):
                print('=========== '+ str(i) +' ===========')
                model(protein,descriptor,i)