#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/4/16 15:01
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : 绘制roc曲线.py

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt  

name = input('请输入文件名：')
data = pd.read_table(name+'.csv',delimiter=',')

y_test = data.iloc[:,0]
y_score = data.iloc[:,1]
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
auc = metrics.auc(fpr,tpr)

roc = plt.figure() 
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot(fpr,tpr,linewidth=2,label='ROC')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,1.0)
plt.ylim(0,1.05)
plt.legend(loc=4)
plt.show()
roc.savefig(name+'.pdf')
with open(name+'.txt','a+') as f:
    f.write('auc: ' + str(auc))