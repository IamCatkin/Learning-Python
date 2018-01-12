#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/7 16:31
# @Author  : Catkin
# @File    : predict.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

#########   遍历目录下所有文件 ########
descriptor = ["cats","maccs","moe2d"]
path = ".\\MODEL\\"
path1 = ".\\TEST\\"
cycles = 20

#########   读取文件1   ############
for d in descriptor:
    header = pd.DataFrame()
    calculator = pd.DataFrame()
    cycle = 0
    filename = ".\\results\\"+"eef2k_"+d
    while cycle < cycles:
        positive = pd.read_table(path+d+"\\eef2k_"+d+"_positive.csv",delimiter=',')
        negative = pd.read_table(path+d+"\\eef2k_"+d+"_negative.csv",delimiter=',')
        random_negative = negative.sample(n=len(positive.index))
        complain = pd.concat([positive,random_negative])
        X = complain.iloc[:,10:len(complain.columns)]
        y = complain.iloc[:,0]
        positive_label = int(complain.iloc[0,0])
        pre_data_0 = pd.read_table(path1+'fda_1430_w-800_1357_wash_'+d+'.csv',delimiter=',',na_values='NaN')
        pre_data = pre_data_0.iloc[:,17:len(pre_data_0.columns)] 
    
#########  划分训练集与测试集 #########
        X_train,X_,y_train,y_ = train_test_split(X,y,test_size=0.0)

#########   使用随机森林算法进行预测 ########
        rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
        answer_rf = rf.predict(pre_data)

#########   评价结果　　　#################
        chance = rf.predict_proba(pre_data)[:,1]

########## 存数据   ##############
        header['id'] = pre_data_0['ID']
        header['smiles'] = pre_data_0['mol']
        calculator['chance-'+str(cycle+1)] = chance
        cycle += 1
    calculator['average'] = calculator.mean(axis=1)
    results = pd.concat([header,calculator],axis=1)
    results.to_csv(filename+'.csv',index=False)