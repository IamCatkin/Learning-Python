#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/24 12:10
# @Author  : Catkin
# @File    : datasets.py
import sys
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

#########  进度条实现  #######
class ShowProcess():
    i = 0 
    max_steps = 0 
    max_arrow = 50 

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) 
        num_line = self.max_arrow - num_arrow 
        percent = self.i * 100.0 / self.max_steps 
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' 
        sys.stdout.write(process_bar) 
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        print(words)
        self.i = 0

#########   初始条件  ########
descriptors = ["cats","maccs","moe2d"]
positive_label = 1
cycles = 20
amount = 250
content_all = pd.read_table("20170609chembl_bindingdb_sum.csv",delimiter=',')
content_filter = content_all[content_all["total"]>=amount]
unis = content_filter["uni"]
max_steps = len(content_filter.index)
process_bar = ShowProcess(max_steps)

#########    读文件   ###############
results = pd.DataFrame()
for uni in unis:
    single = pd.DataFrame()
    single["pname"] = None
    single.loc[0] = uni
    for descriptor in descriptors:
        cycle = 0
        file = ".\\"+descriptor+"\\"+uni
        result = pd.DataFrame()
        result["acc"] = None
        result["spe"] = None
        result["sen"] = None
        result["auc"] = None
        while cycle < cycles:
            data = pd.read_table(file+".txt",delimiter='\t',na_values='NaN')
            data['label'] = 0
            activity = data.iloc[:,3]
            median = activity.median()
            for i in range(len(data.index)):
                if data.iloc[i,3]<median:
                    data.iloc[i,len(data.columns)-1]=1
                else:
                    data.iloc[i,len(data.columns)-1]=0
            X = data.iloc[:,7:len(data.columns)-1]
            y = data.iloc[:,len(data.columns)-1]
    
########    使用中位值补全缺失值     #############
            X.fillna(value=X.median(),inplace=True)   
            
##########  划分训练集与测试集 #########
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

##########   使用随机森林算法进行预测 ########
            rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
            answer_rf = rf.predict(X_test)

##########   评价结果　　　#################
            y_score = rf.predict_proba(X_test)[:,1]
            fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=positive_label)
            confusion = metrics.confusion_matrix(y_test,answer_rf)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            accuracy = metrics.accuracy_score(y_test,answer_rf)
            specificity = TN / (TN+FP)
            sensitivity = TP / (TP+FN)
            auc = metrics.auc(fpr,tpr)

#########   存数据  #############
            result.loc[cycle] = [accuracy,specificity,sensitivity,auc]
            cycle += 1
        single[descriptor+"_acc_mean"] = result.iloc[:,0].mean()
        single[descriptor+"_acc_var"] = result.iloc[:,0].var()
        single[descriptor+"_spe_mean"] = result.iloc[:,1].mean()
        single[descriptor+"_spe_var"] = result.iloc[:,1].var()
        single[descriptor+"_sen_mean"] = result.iloc[:,2].mean()
        single[descriptor+"_sen_var"] = result.iloc[:,2].var()
        single[descriptor+"_auc_mean"] = result.iloc[:,3].mean()
        single[descriptor+"_auc_var"] = result.iloc[:,3].var()
    results = pd.concat([results,single])
    results.to_csv('results.csv',index=False)
    process_bar.show_process()
process_bar.close()