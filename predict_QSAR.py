# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

#########   遍历目录下所有文件 ########
path = ".\\moe2d\\"
files_0 = os.listdir(path)  
files = []
for f in files_0:
    if f != 'Neferine.txt':
        files.append(f)

#########  建立结果文档  #########

filename = ".\\results\\"+"results.csv"
with open(filename,'a+') as f:
    f.write("pname"+','+"Pre"+','+"pos_chance"+'\n')
    f.close()

#########   读取文件   ############
for file in files:
    data = pd.read_table(path+file,delimiter='\t',na_values='NaN')
    data['label'] = 0
    activity = data.iloc[:,3]
    median = activity.median()
    if median > 10000:
        median = 10000
    for i in range(len(data.index)):
        if data.iloc[i,3]<median:
            data.iloc[i,len(data.columns)-1]=1
        else:
            data.iloc[i,len(data.columns)-1]=0
    X = data.iloc[:,7:len(data.columns)-1]
    y = data.iloc[:,len(data.columns)-1]
    pre_data_0 = pd.read_table(path+'Neferine.txt',delimiter='\t',na_values='NaN')
    pre_data = pre_data_0.iloc[:,1:len(pre_data_0.columns)]

######## 使用中位值补全缺失值，然后将数据补全  #############
    imp = Imputer(missing_values='NaN',strategy='median',axis=0)
    imp.fit(X)
    X0 = imp.transform(X)     
    X = pd.DataFrame(X0,index=X.index,columns=X.columns)   # 从ndarray转回DataFrame

##########  划分训练集与测试集 #########
    X_train,X_,y_train,y_ = train_test_split(X,y,test_size=0.0)

##########   使用随机森林算法进行预测 ########
    rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
    answer_rf = rf.predict(pre_data)

##########   评价结果　　　#################
    chance = rf.predict_proba(pre_data)[0,1]
        
########## 存数据   ############## 
    with open(filename,'a+') as f:
        f.write(file[0:6]+','+str(answer_rf[0])+','+str(chance)+'\n')
        f.close()