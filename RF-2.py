# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import matplotlib.pyplot as plt  


seq = 1
while seq < 23:
    cycle = 0
    filename = ".\\results\\"+str(seq)+".csv"
    with open(filename,'a+') as f:
        f.write('ACC'+','+'SPE'+','+'SEN'+','+'AUC'+'\n')
        f.close()
    while cycle < 50:
#########   读取文件   ############
        path =r"D:\python\txt\\"
        data = pd.read_table(path+str(seq)+'.txt',delimiter='\t')
        positive = data.iloc[0:len(data.index),3:len(data.columns)]
        path_0 = r'D:\python\txt\label0.txt'
        data_0 = pd.read_table(path_0,delimiter='\t')
        negative = data_0.iloc[0:len(data_0.index),3:len(data.columns)]
        random_negative = negative.sample(n=len(data.index))
        complain = pd.concat([positive,random_negative])
        X = complain.iloc[:,1:len(complain.columns)]
        y = complain.iloc[:,0]
        positive_label = int(complain.iloc[0,0])

#########  划分训练集与测试集 #########
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#########   使用随机森林算法进行预测 ########
        rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
        answer_rf = rf.predict(X_test)

#########   评价结果　　　#################
        y_score = rf.predict_proba(X_test)[:,1]
        fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=positive_label)
        confusion=metrics.confusion_matrix(y_test,answer_rf)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        auc = metrics.auc(fpr,tpr)
        accuracy = metrics.accuracy_score(y_test,answer_rf)
        sensitivity = TP / (TP+FN)
        specificity = TN / (TN+FP)
        #print(metrics.classification_report(y_test, answer_rf)) #生成评价报告


############ 绘制ROC曲线 ############
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.plot(fpr,tpr,linewidth=2,label='ROC')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.xlim(0,1.0)
        plt.ylim(0,1.05)
        plt.legend(loc=4)#图例的位置
        #plt.show()

########## 需要的数据 ##########
        print('The auc is: %.3f' %auc)
        print('The accuracy is: %.3f' %accuracy)
        print('The sensitivity is: %.3f' %sensitivity)
        print('The specificity is: %.3f' %specificity)

########## 存数据   ##############
        with open(filename,'a+') as f:
            f.write('{:.3f}'.format(float(accuracy))+','+'{:.3f}'.format(float(specificity))+','+'{:.3f}'.format(float(sensitivity))+','+'{:.3f}'.format(float(auc))+'\n')
            f.close()
            cycle += 1
    seq += 1
    
