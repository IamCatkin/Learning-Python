#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/7 16:31
# @Author  : Catkin
# @File    : predict.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

#########   遍历目录下所有文件 ########
series = ["series1","series2","all data","literature data"]
path = ".\\txt\\"

#########   读取文件系列1   ############
for s in series:
    if s == "series1":
        files_1 = os.listdir(path+s)  
        files = []
        for f in files_1:
            if f != 'hepdata0_1038_2d.txt':
                files.append(f)

        for file in files:
            cycle = 0
            filename = ".\\results\\"+s+"\\"+file[:-4]
            with open(filename+".csv",'a+') as f1:
                f1.write("Pre"+','+"pos_chance"+'\n')
                f1.close()
            while cycle < 1:
                data = pd.read_table(path+s+"\\"+file,delimiter='\t')
                positive = data.iloc[0:len(data.index),3:len(data.columns)]
                data_0 = pd.read_table(path+s+"\\"+"hepdata0_1038_2d.txt",delimiter='\t')
                negative = data_0.iloc[0:len(data_0.index),3:len(data.columns)]
                random_negative = negative.sample(n=len(data.index))
                complain = pd.concat([positive,random_negative])
                X = complain.iloc[:,1:len(complain.columns)]
                y = complain.iloc[:,0]
                positive_label = int(complain.iloc[0,0])
                pre_data_0 = pd.read_table(path+'database_test.txt',delimiter='\t',na_values='NaN')
                pre_data_0 = pre_data_0.loc[pre_data_0["lable"]==positive_label]
                pre_data = pre_data_0.iloc[:,4:len(pre_data_0.columns)]
        
#########  划分训练集与测试集 #########
                X_train,X_,y_train,y_ = train_test_split(X,y,test_size=0.0)

#########   使用随机森林算法进行预测 ########
                rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
                answer_rf = rf.predict(pre_data)

#########   评价结果　　　#################
                chance = rf.predict_proba(pre_data)[:,1]

########## 存数据   ##############
                with open(filename+".csv",'a+') as f1:
                    for r,c in zip(answer_rf,chance):
                        f1.write(str(r)+','+str(c)+"\n")
                    f1.close()
                cycle += 1
                
#########   读取文件系列2   ############
    elif s == "series2":
        files_2 = os.listdir(path+s)  
        files = []
        for f in files_2:
            if f != 'hepdata0_1038_2d.txt':
                files.append(f)

        for file in files:
            cycle = 0
            filename = ".\\results\\"+s+"\\"+file[:-4]
            with open(filename+".csv",'a+') as f2:
                f2.write("Pre"+','+"pos_chance"+'\n')
                f2.close()
            while cycle < 1:
                data = pd.read_table(path+s+"\\"+file,delimiter='\t')
                positive = data.iloc[0:len(data.index),3:len(data.columns)]
                data_0 = pd.read_table(path+s+"\\"+"hepdata0_1038_2d.txt",delimiter='\t')
                negative = data_0.iloc[0:len(data_0.index),3:len(data.columns)]
                random_negative = negative.sample(n=len(data.index))
                complain = pd.concat([positive,random_negative])
                X = complain.iloc[:,1:len(complain.columns)]
                y = complain.iloc[:,0]
                positive_label = int(complain.iloc[0,0])
                pre_data_0 = pd.read_table(path+'database_test.txt',delimiter='\t',na_values='NaN')
                pre_data_0 = pre_data_0.loc[pre_data_0["lable"]==positive_label]
                pre_data = pre_data_0.iloc[:,4:len(pre_data_0.columns)]
        
#########  划分训练集与测试集 #########
                X_train,X_,y_train,y_ = train_test_split(X,y,test_size=0.0)

#########   使用随机森林算法进行预测 ########
                rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
                answer_rf = rf.predict(pre_data)

#########   评价结果　　　#################
                chance = rf.predict_proba(pre_data)[:,1]

########## 存数据   ##############
                with open(filename+".csv",'a+') as f2:
                    for r,c in zip(answer_rf,chance):
                        f2.write(str(r)+','+str(c)+"\n")
                    f2.close()
                cycle += 1
                
#########   读取文件系列3   ############
    elif s == "all data":
        cycle = 0
        file = "all_data.txt"
        filename = filename = ".\\results\\"+s+"\\"+"all_data"
        with open(filename+".csv",'a+') as f3:
                f3.write("Pre"+','+"pos_chance"+'\n')
                f3.close()
        while cycle < 1:       
            data = pd.read_table(path+s+"\\"+file,delimiter='\t')
            X = data.iloc[:,4:len(data.columns)]
            y = data.iloc[:,3]
            pre_data_0 = pd.read_table(path+'database_test.txt',delimiter='\t',na_values='NaN')
            pre_data = pre_data_0.iloc[:,4:len(pre_data_0.columns)]
            
#########  划分训练集与测试集 #########
            X_train,X_,y_train,y_ = train_test_split(X,y,test_size=0.0)

#########   使用随机森林算法进行预测 ########
            rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
            answer_rf = rf.predict(pre_data)

#########   评价结果　　　#################
            chance = rf.predict_proba(pre_data)[:,1]

########## 存数据   ##############
            with open(filename+".csv",'a+') as f3:
                for r,c in zip(answer_rf,chance):
                    f3.write(str(r)+','+str(c)+"\n")
                f3.close()
            cycle += 1       
            
#########   读取文件系列4   ############
    else:        
        cycle = 0
        file = "literdata_2d.txt"
        filename = filename = ".\\results\\"+s+"\\"+"literdata_2d"
        with open(filename+".csv",'a+') as f4:
                f4.write("Pre"+','+"pos_chance"+'\n')
                f4.close()
        while cycle < 1:       
            data = pd.read_table(path+s+"\\"+file,delimiter='\t')
            X = data.iloc[:,4:len(data.columns)]
            y = data.iloc[:,3]
            pre_data_0 = pd.read_table(path+'database_test.txt',delimiter='\t',na_values='NaN')
            pre_data = pre_data_0.iloc[:,4:len(pre_data_0.columns)]
            
#########  划分训练集与测试集 #########
            X_train,X_,y_train,y_ = train_test_split(X,y,test_size=0.0)

#########   使用随机森林算法进行预测 ########
            rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
            answer_rf = rf.predict(pre_data)

#########   评价结果　　　#################
            chance = rf.predict_proba(pre_data)[:,1]

########## 存数据   ##############
            with open(filename+".csv",'a+') as f4:
                for r,c in zip(answer_rf,chance):
                    f4.write(str(r)+','+str(c)+"\n")
                f4.close()
            cycle += 1               