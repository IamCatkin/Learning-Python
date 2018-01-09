#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 21:58
# @Author  : Catkin
# @File    : eef2k.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import matplotlib.pyplot as plt  

#########   遍历目录下所有文件 ########
descriptor = ["cats","maccs","moe2d"]
path = ".\\MODEL\\"

#########   读取文件1   ############
for d in descriptor:
    if d == "cats":
        cycle = 0
        filename = ".\\results\\"+"eef2k_cats"+".csv"
        with open(filename,'a+') as f:
            f.write('ACC'+','+'SPE'+','+'SEN'+','+'AUC'+'\n')
            f.close()
        while cycle < 50:
            positive = pd.read_table(path+d+"\\eef2k_cats_positive.csv",delimiter=',')
            negative = pd.read_table(path+d+"\\eef2k_cats_negative.csv",delimiter=',')
            random_negative = negative.sample(n=len(positive.index))
            complain = pd.concat([positive,random_negative])
            X = complain.iloc[:,10:len(complain.columns)]
            y = complain.iloc[:,0]
            positive_label = int(complain.iloc[0,0])
        
##########  划分训练集与测试集 #########
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

##########   使用随机森林算法进行预测 ########
            rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
            answer_rf = rf.predict(X_test)

##########   评价结果　　　#################
            y_score = rf.predict_proba(X_test)[:,1]
            fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
            confusion = metrics.confusion_matrix(y_test,answer_rf)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            auc = metrics.auc(fpr,tpr)
            accuracy = metrics.accuracy_score(y_test,answer_rf)
            sensitivity = TP / (TP+FN)
            specificity = TN / (TN+FP)

############ 绘制ROC曲线 ############
            plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
            plt.plot(fpr,tpr,linewidth=2,label='ROC')
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.xlim(0,1.0)
            plt.ylim(0,1.05)
            plt.legend(loc=4)#图例的位置
            plt.show()

########## 需要的数据 ##########
            print('The accuracy is: %.3f' %accuracy)
            print('The specificity is: %.3f' %specificity)
            print('The sensitivity is: %.3f' %sensitivity)
            print('The auc is: %.3f' %auc)
            
########## 存数据   ############## 
            with open(filename,'a+') as f:
                f.write('{:.3f}'.format(float(accuracy))+','+'{:.3f}'.format(float(specificity))+','+'{:.3f}'.format(float(sensitivity))+','+'{:.3f}'.format(float(auc))+'\n')
                f.close()
            cycle += 1
            
#########   读取文件2   ############
    elif d == "maccs":
        cycle = 0
        filename = ".\\results\\"+"eef2k_maccs"+".csv"
        with open(filename,'a+') as f:
            f.write('ACC'+','+'SPE'+','+'SEN'+','+'AUC'+'\n')
            f.close()
        while cycle < 50:
            positive = pd.read_table(path+d+"\\eef2k_maccs_positive.csv",delimiter=',')
            negative = pd.read_table(path+d+"\\eef2k_maccs_negative.csv",delimiter=',')
            random_negative = negative.sample(n=len(positive.index))
            complain = pd.concat([positive,random_negative])
            X = complain.iloc[:,10:len(complain.columns)]
            y = complain.iloc[:,0]
            positive_label = int(complain.iloc[0,0])
        
##########  划分训练集与测试集 #########
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

##########   使用随机森林算法进行预测 ########
            rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
            answer_rf = rf.predict(X_test)

##########   评价结果　　　#################
            y_score = rf.predict_proba(X_test)[:,1]
            fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
            confusion = metrics.confusion_matrix(y_test,answer_rf)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            auc = metrics.auc(fpr,tpr)
            accuracy = metrics.accuracy_score(y_test,answer_rf)
            sensitivity = TP / (TP+FN)
            specificity = TN / (TN+FP)

############ 绘制ROC曲线 ############
            plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
            plt.plot(fpr,tpr,linewidth=2,label='ROC')
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.xlim(0,1.0)
            plt.ylim(0,1.05)
            plt.legend(loc=4)#图例的位置
            plt.show()

########## 需要的数据 ##########
            print('The accuracy is: %.3f' %accuracy)
            print('The specificity is: %.3f' %specificity)
            print('The sensitivity is: %.3f' %sensitivity)
            print('The auc is: %.3f' %auc)
            
########## 存数据   ############## 
            with open(filename,'a+') as f:
                f.write('{:.3f}'.format(float(accuracy))+','+'{:.3f}'.format(float(specificity))+','+'{:.3f}'.format(float(sensitivity))+','+'{:.3f}'.format(float(auc))+'\n')
                f.close()
            cycle += 1            
            
#########   读取文件3   ############
    else:
        cycle = 0
        filename = ".\\results\\"+"eef2k_moe2d"+".csv"
        with open(filename,'a+') as f:
            f.write('ACC'+','+'SPE'+','+'SEN'+','+'AUC'+'\n')
            f.close()
        while cycle < 50:
            positive = pd.read_table(path+d+"\\eef2k_moe2d_positive.csv",delimiter=',')
            negative = pd.read_table(path+d+"\\eef2k_moe2d_negative.csv",delimiter=',')
            random_negative = negative.sample(n=len(positive.index))
            complain = pd.concat([positive,random_negative])
            X = complain.iloc[:,10:len(complain.columns)]
            y = complain.iloc[:,0]
            positive_label = int(complain.iloc[0,0])
        
##########  划分训练集与测试集 #########
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

##########   使用随机森林算法进行预测 ########
            rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
            answer_rf = rf.predict(X_test)

##########   评价结果　　　#################
            y_score = rf.predict_proba(X_test)[:,1]
            fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
            confusion = metrics.confusion_matrix(y_test,answer_rf)
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            auc = metrics.auc(fpr,tpr)
            accuracy = metrics.accuracy_score(y_test,answer_rf)
            sensitivity = TP / (TP+FN)
            specificity = TN / (TN+FP)

############ 绘制ROC曲线 ############
            plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
            plt.plot(fpr,tpr,linewidth=2,label='ROC')
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.xlim(0,1.0)
            plt.ylim(0,1.05)
            plt.legend(loc=4)#图例的位置
            plt.show()

########## 需要的数据 ##########
            print('The accuracy is: %.3f' %accuracy)
            print('The specificity is: %.3f' %specificity)
            print('The sensitivity is: %.3f' %sensitivity)
            print('The auc is: %.3f' %auc)
            
########## 存数据   ############## 
            with open(filename,'a+') as f:
                f.write('{:.3f}'.format(float(accuracy))+','+'{:.3f}'.format(float(specificity))+','+'{:.3f}'.format(float(sensitivity))+','+'{:.3f}'.format(float(auc))+'\n')
                f.close()
            cycle += 1                        
