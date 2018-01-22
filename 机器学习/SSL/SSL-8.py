#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/17 15:21
# @Author  : Catkin
# @File    : SSL.py

###################################################
#添加大概率预测为正/负至模型(单一描述符&双方法&改变标签)
###################################################
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics
uni = "P35968"
path = ".\\models\\"
pre_results = pd.DataFrame()
prediction = pd.DataFrame()
prediction['rate']=None

######## 参数 #########
cycles = 100
label_rate = 0.05
addition = 10
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
    X_1,X_test,y_1,y_test = train_test_split(X_0,y_0,test_size=0.2,random_state=332)
    X_2,X_3,y_2,y_3 = train_test_split(X_1,y_1,test_size=1-label_rate,random_state=323)

##############  整体预测与交互检验  ###########
#    scores_all = cross_val_score(RandomForestClassifier(n_estimators=500), X_1, y_1, cv=5, scoring='accuracy')
#    score_all_mean =scores_all.mean()
#    print(d+'整体5折交互检验:'+str(score_all_mean))
#    rf_all = RandomForestClassifier(n_estimators=500).fit(X_1,y_1)
#    answer_rf_all = rf_all.predict(X_test)
#    accuracy_all = metrics.accuracy_score(y_test,answer_rf_all)
#    print(d+'整体预测:'+str(accuracy_all))
################################################
    
    return data,X_2,y_2,X_3,y_3,X_test,y_test

def rfmodel(d,X_2,y_2,X_3,y_3,X_test,y_test):
    X_3_copy = X_3.copy(deep=True)
    X_3_copy['chance']=0
    index = 0    
    
########## k折交叉验证 ###########################
    scores = cross_val_score(RandomForestClassifier(n_estimators=500), X_2, y_2, cv=5, scoring='accuracy')
    score_mean =scores.mean()
    print(d+'5折交互检验:'+str(score_mean))
#################################################
    
    rf = RandomForestClassifier(n_estimators=500).fit(X_2,y_2)

################ 预测测试集 ################   
    answer_rf = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,answer_rf)
    print(d+'预测:'+str(accuracy))
###############################################
    
    chance = rf.predict_proba(X_3)[:,1]
    for c in chance:
        X_3_copy.iloc[index,len(X_3_copy.columns)-1]=c
        index += 1
    chance_que = X_3_copy.iloc[:,len(X_3_copy.columns)-1]
    return chance_que

def gnbmodel(d,X_2,y_2,X_3,y_3,X_test,y_test):
    X_3_copy = X_3.copy(deep=True)
    X_3_copy['chance']=0
    index = 0    
    
########## k折交叉验证 ###########################
    scores = cross_val_score(GaussianNB(), X_2, y_2, cv=5, scoring='accuracy')
    score_mean =scores.mean()
    print(d+'5折交互检验:'+str(score_mean))
#################################################
    
    gnb = GaussianNB().fit(X_2,y_2)

################ 预测测试集 ################   
    answer_gnb = gnb.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,answer_gnb)
    print(d+'预测:'+str(accuracy))
###############################################
    
    chance = gnb.predict_proba(X_3)[:,1]
    for c in chance:
        X_3_copy.iloc[index,len(X_3_copy.columns)-1]=c
        index += 1
    chance_que = X_3_copy.iloc[:,len(X_3_copy.columns)-1]
    return chance_que

def svcmodel(d,X_2,y_2,X_3,y_3,X_test,y_test):
    X_3_copy = X_3.copy(deep=True)
    X_3_copy['chance']=0
    index = 0    
    
########## k折交叉验证 ###########################
    scores = cross_val_score(SVC(), X_2, y_2, cv=5, scoring='accuracy')
    score_mean =scores.mean()
    print(d+'5折交互检验:'+str(score_mean))
#################################################
    
    svc = SVC(probability=True).fit(X_2,y_2)

################ 预测测试集 ################   
    answer_svc = svc.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,answer_svc)
    print(d+'预测:'+str(accuracy))
###############################################
    
    chance = svc.predict_proba(X_3)[:,1]
    for c in chance:
        X_3_copy.iloc[index,len(X_3_copy.columns)-1]=c
        index += 1
    chance_que = X_3_copy.iloc[:,len(X_3_copy.columns)-1]
    return chance_que

def getindex(chance_que):
    positive = chance_que.sort_values(ascending=False).head(addition)
    negative = chance_que.sort_values(ascending=True).head(addition)
    return positive,negative

def getdropindex(a,b,c,d):
    one = list(set(a).union(set(b)))
    two = list(set(c).union(set(d)))
    three = list(set(one).union(set(two)))
    return three

def update(data,X_2,y_2,X_3,y_3,added_positive,added_negative,droplist):

############### add ################
    for p in added_positive.index:
        X_2 = pd.concat([X_2,data.iloc[p:p+1,7:len(data.columns)-1]])
        y_2.loc[p] = 1
    for n in added_negative.index:
        X_2 = pd.concat([X_2,data.iloc[n:n+1,7:len(data.columns)-1]])
        y_2.loc[n] = 0
######################################

############## drop ###################    
    for item in droplist:
        X_3.drop(item,inplace=True)
        y_3.drop(item,inplace=True)
    return X_2,y_2,X_3,y_3

def preresults(d,cycle,data,added_positive,added_negative):    
    pre_results[d+'_pre_1_'+str(cycle+1)]=None
    pre_results[d+'_pre_0_'+str(cycle+1)]=None
    count_1 = 0
    count_2 = 0
    for px in added_positive.index:
        pre_results.loc[count_1,d+'_pre_1_'+str(cycle+1)] = data.iloc[px,len(data.columns)-1]
        count_1 += 1
    for nx in added_negative.index:
        pre_results.loc[count_2,d+'_pre_0_'+str(cycle+1)] = data.iloc[nx,len(data.columns)-1]
        count_2 += 1
    prediction.loc[cycle+1] = (pre_results[d+'_pre_1_'+str(cycle+1)].sum()/addition + (1-pre_results[d+'_pre_0_'+str(cycle+1)].sum()/addition))/2       
    return pre_results,prediction

if __name__ == '__main__':
    cycle = 0
    data_cats1,X_cats1,y_cats1,X_cats1_3,y_cats1_3,X_cats1_test,y_cats1_test = read("cats")
    data_cats2,X_cats2,y_cats2,X_cats2_3,y_cats2_3,X_cats2_test,y_cats2_test = read("cats")
    
    while cycle < cycles:
        print("======第"+str(cycle+1)+"次======")
        cats1_chance = rfmodel("cats1",X_cats1,y_cats1,X_cats1_3,y_cats1_3,X_cats1_test,y_cats1_test)
        cats2_chance = svcmodel("cats2",X_cats2,y_cats2,X_cats2_3,y_cats2_3,X_cats2_test,y_cats2_test)
        
        cats1_added_positive,cats1_added_negative = getindex(cats2_chance)
        cats2_added_positive,cats2_added_negative = getindex(cats1_chance)
        
        droplist = getdropindex(cats1_added_positive.index,cats1_added_negative.index,cats2_added_positive.index,cats2_added_negative.index)
        
        cats1_pre_results,cats1_prediction = preresults("cats1",cycle,data_cats1,cats2_added_positive,cats2_added_negative)
        cats2_pre_results,cats2_prediction = preresults("cats2",cycle,data_cats2,cats1_added_positive,cats1_added_negative)
  
        X_cats1,y_cats1,X_cats1_3,y_cats1_3 = update(data_cats1,X_cats1,y_cats1,X_cats1_3,y_cats1_3,cats1_added_positive,cats1_added_negative,droplist)
        X_cats2,y_cats2,X_cats2_3,y_cats2_3 = update(data_cats2,X_cats2,y_cats2,X_cats2_3,y_cats2_3,cats2_added_positive,cats2_added_negative,droplist)
        print("================\n")
        
        cycle += 1
        