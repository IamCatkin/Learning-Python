import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics

uni = "P35968"
path = ".\\models\\"

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
    X_train,X_test,y_train,y_test = train_test_split(X_0,y_0,test_size=0.2,random_state=7)

##############  整体预测与交互检验  ###########
#    scores_all = cross_val_score(RandomForestClassifier(n_estimators=500), X_1, y_1, cv=5, scoring='accuracy')
#    score_all_mean =scores_all.mean()
#    print(d+'5折交互检验:'+str(score_all_mean))
#    rf_all = RandomForestClassifier(n_estimators=500).fit(X_1,y_1)
#    answer_rf_all = rf_all.predict(X_test)
#    accuracy_all = metrics.accuracy_score(y_test,answer_rf_all)
#    print(d+'整体预测:'+str(accuracy_all))
################################################
    
    return X_train,y_train,X_test,y_test

X_train,y_train,X_test,y_test = read("maccs")
rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)

################ 预测测试集 ################   
answer_rf = rf.predict(X_test)
y_score = rf.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
auc = metrics.auc(fpr,tpr)
accuracy = metrics.accuracy_score(y_test,answer_rf)
print('acc预测:'+str(accuracy))
print('auc预测:'+str(auc))