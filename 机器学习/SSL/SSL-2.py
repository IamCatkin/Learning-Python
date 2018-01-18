#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/17 15:21
# @Author  : Catkin
# @File    : SSL.py

#################################################
#添加其他描叙符预测为正/负且本描叙符预测相反的至模型
###############################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics
uni = "P35968"
path = ".\\models\\"

######## 参数 #########
cycles = 5
label_rate = 0.05
addition = 50
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
    X_0,X_,y_0,y_ = train_test_split(X_0,y_0,test_size=0.0,random_state=3421)
    X_1,X_test,y_1,y_test = train_test_split(X_0,y_0,test_size=0.2,random_state=1257)
    X_2,X_3,y_2,y_3 = train_test_split(X_1,y_1,test_size=1-label_rate,random_state=11)

##############  整体预测与交互检验  ###########
#    scores_all = cross_val_score(RandomForestClassifier(n_estimators=500), X_1, y_1, cv=5, scoring='accuracy')
#    score_all_mean =scores_all.mean()
#    print(d+'5折交互检验:'+str(score_all_mean))
#    rf_all = RandomForestClassifier(n_estimators=500).fit(X_1,y_1)
#    answer_rf_all = rf_all.predict(X_test)
#    accuracy_all = metrics.accuracy_score(y_test,answer_rf_all)
#    print(d+'整体预测:'+str(accuracy_all))
################################################
    
    return data,X_2,y_2,X_3,y_3,X_test,y_test

def model(d,X_2,y_2,X_3,y_3,X_test,y_test):
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

def getindex(c,c1,c2):
    positive = []
    negative = []
    sorted_chance = pd.DataFrame()
    sorted_chance['average']=(c1+c2)/2
    positive_0 = sorted_chance.sort_values(['average'], ascending=False)
    negative_0 = sorted_chance.sort_values(['average'], ascending=True)
    for p_index in positive_0.index:
        if c.loc[p_index]<0.5:
            positive.append(p_index)
    positive=positive[:addition]
    for n_index in negative_0.index:
        if c.loc[n_index]>=0.5:
            negative.append(n_index)
    negative=negative[:addition]
                
    return positive,negative

def getdropindex(a,b,c,d,e,f):
    one = list(set(a).union(set(b)))
    two = list(set(c).union(set(d)))
    three = list(set(e).union(set(f)))
    four = list(set(one).union(set(two)))
    five = list(set(three).union(set(four)))
    return five
    
def update(data,X_2,y_2,X_3,y_3,added_positive,added_negative,droplist):

############### add ################
    for p in added_positive:
        X_2 = pd.concat([X_2,data.iloc[p:p+1,7:len(data.columns)-1]])
        y_2.loc[p] = 1
    for n in cats_added_negative:
        X_2 = pd.concat([X_2,data.iloc[n:n+1,7:len(data.columns)-1]])
        y_2.loc[n] = 0
######################################

############## drop ###################    
    for item in droplist:
        X_3.drop(item,inplace=True)
        y_3.drop(item,inplace=True)
    return X_2,y_2,X_3,y_3
    
if __name__ == '__main__':
    cycle = 0
    data_cats,X_cats,y_cats,X_cats_3,y_cats_3,X_cats_test,y_cats_test = read("cats")
    data_maccs,X_maccs,y_maccs,X_maccs_3,y_maccs_3,X_maccs_test,y_maccs_test = read("maccs")
    data_moe2d,X_moe2d,y_moe2d,X_moe2d_3,y_moe2d_3,X_moe2d_test,y_moe2d_test = read("moe2d")
    
    while cycle < cycles:
        print("======第"+str(cycle+1)+"次======")
        cats_chance = model("cats",X_cats,y_cats,X_cats_3,y_cats_3,X_cats_test,y_cats_test)
        maccs_chance = model("maccs",X_maccs,y_maccs,X_maccs_3,y_maccs_3,X_maccs_test,y_maccs_test)    
        moe2d_chance = model("moe2d",X_moe2d,y_moe2d,X_moe2d_3,y_moe2d_3,X_moe2d_test,y_moe2d_test)
        
        cats_added_positive,cats_added_negative = getindex(cats_chance,maccs_chance,moe2d_chance)
        maccs_added_positive,maccs_added_negative = getindex(maccs_chance,cats_chance,moe2d_chance)
        moe2d_added_positive,moe2d_added_negative = getindex(moe2d_chance,cats_chance,maccs_chance)
        
        droplist = getdropindex(cats_added_positive,cats_added_negative,maccs_added_positive,maccs_added_negative,moe2d_added_positive,moe2d_added_negative)
           
        X_cats,y_cats,X_cats_3,y_cats_3 = update(data_cats,X_cats,y_cats,X_cats_3,y_cats_3,cats_added_positive,cats_added_negative,droplist)
        X_maccs,y_maccs,X_maccs_3,y_maccs_3 = update(data_maccs,X_maccs,y_maccs,X_maccs_3,y_maccs_3,maccs_added_positive,maccs_added_negative,droplist)
        X_moe2d,y_moe2d,X_moe2d_3,y_moe2d_3 = update(data_moe2d,X_moe2d,y_moe2d,X_moe2d_3,y_moe2d_3,moe2d_added_positive,moe2d_added_negative,droplist)
        print("================\n")
        
        cycle += 1