#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/17 15:21
# @Author  : Catkin
# @File    : SSL.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#################################
#添加本身即为正/负至模型(单一描述符)
##################################

uni = "P35968"
path = ".\\models\\"

######## 参数 #########
cycles = 10
label_rate = 0.05
addition = 100
#######################

def read(d):
    data = pd.read_table(path+uni+"_"+d+".txt",delimiter='\t')
    data['label'] = 0
    for i in range(len(data.index)):
        if data.iloc[i,3]<1000:
            data.iloc[i,len(data.columns)-1]=1
        else:
            data.iloc[i,len(data.columns)-1]=0
    X_0 = data.iloc[:,7:len(data.columns)-1]
    y_0 = data.iloc[:,len(data.columns)-1]    
    X_1,X_test,y_1,y_test = train_test_split(X_0,y_0,test_size=0.2,random_state=None)
    X_2,X_3,y_2,y_3 = train_test_split(X_1,y_1,test_size=1-label_rate,random_state=None)

#############  整体预测与交互检验  ###########
    scores_all = cross_val_score(RandomForestClassifier(n_estimators=500), X_1, y_1, cv=5, scoring='accuracy')
    score_all_mean =scores_all.mean()
    print(d+'整体5折交互检验:'+str(score_all_mean))
    rf_all = RandomForestClassifier(n_estimators=500).fit(X_1,y_1)
    answer_rf_all = rf_all.predict(X_test)
    accuracy_all = metrics.accuracy_score(y_test,answer_rf_all)
    print(d+'整体预测:'+str(accuracy_all))
###############################################
    
    return data,X_2,y_2,X_3,y_3,X_test,y_test

def model(d,X_2,y_2,X_3,y_3,X_test,y_test):
    X_3_copy = X_3.copy(deep=True)
    X_3_copy['chance']=0
    
########## k折交叉验证 ###########################
    scores = cross_val_score(RandomForestClassifier(n_estimators=500), X_2, y_2, cv=5, scoring='accuracy')
    score_mean =scores.mean()
    print(d+'5折交互检验:'+str(score_mean))    

    rf = RandomForestClassifier(n_estimators=500).fit(X_2,y_2)

################ 预测测试集 ################   
    answer_rf = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,answer_rf)
    print(d+'预测:'+str(accuracy))

def getindex(y_3):
    positive_all = y_3[y_3 == 1]
    negative_all = y_3[y_3 == 0]
    positive = positive_all.sample(n=addition)
    negative = negative_all.sample(n=addition)
    droplist = pd.concat([positive,negative])
    return positive,negative,droplist
    
def update(data,X_2,y_2,X_3,y_3,added_positive,added_negative,droplist):

############### add ################
    for p in added_positive.index:
        X_2 = pd.concat([X_2,data.iloc[p:p+1,7:len(data.columns)-1]])
        y_2.loc[p] = 1
    for n in added_negative.index:
        X_2 = pd.concat([X_2,data.iloc[n:n+1,7:len(data.columns)-1]])
        y_2.loc[n] = 0

############## drop ###################    
    for item in droplist.index:
        X_3.drop(item,inplace=True)
        y_3.drop(item,inplace=True)
    return X_2,y_2,X_3,y_3

if __name__ == '__main__':
    cycle = 0
    data_cats,X_cats,y_cats,X_cats_3,y_cats_3,X_cats_test,y_cats_test = read("cats")
  
    while cycle < cycles:
        print("======第"+str(cycle+1)+"次======")
        model("cats",X_cats,y_cats,X_cats_3,y_cats_3,X_cats_test,y_cats_test)
        cats_added_positive,cats_added_negative,cats_droplist = getindex(y_cats_3)
        X_cats,y_cats,X_cats_3,y_cats_3 = update(data_cats,X_cats,y_cats,X_cats_3,y_cats_3,cats_added_positive,cats_added_negative,cats_droplist)
        print("================\n")
        
        cycle += 1
        
