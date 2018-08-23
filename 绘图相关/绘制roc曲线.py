#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/17 16:01
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : 绘制roc曲线.py
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt  
import seaborn as sns

def getfilenames():
    files = os.listdir()  
    names = set()
    for file in files:
        if '.csv' in file:
            name = file[:-4]
            names.add(name)
    names = list(names)
    return names

def getdata(filenames):
    fprs = dict()
    tprs = dict()
    roc_aucs = dict()
    for filename in filenames:
        data = pd.read_table(filename+'.csv',delimiter=',')
        y_test = data.iloc[:,0]
        y_score = data.iloc[:,1]
        fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
        roc_auc = metrics.auc(fpr,tpr)
        fprs[filename],tprs[filename],roc_aucs[filename] = fpr,tpr,roc_auc
    return fprs,tprs,roc_aucs

def makeroc(fprs,tprs,roc_aucs,filenames):
    f = plt.figure(figsize=(16, 12))
    colors = sns.color_palette()
    for filename, color in zip(filenames, colors):
        plt.plot(fprs[filename], tprs[filename], color=color, lw=2,
                 label=filename+' AUC=%.2f' %roc_aucs[filename])
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=16)  
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.legend(loc="lower right",fontsize=18)
    plt.show()    
    f.savefig('roc.pdf')    
    
if __name__ == '__main__':
    filenames = getfilenames()
    fprs,tprs,roc_aucs = getdata(filenames)
    makeroc(fprs,tprs,roc_aucs,filenames)