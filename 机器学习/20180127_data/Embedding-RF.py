#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Mon May 14 23:11:27 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : Embedding-RF.py
import os
import pandas as pd
import numpy as np
from itertools import chain
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
#########     参数     ###########
path = '.\\data\\'
descriptors = ["cats","ecfp4","maccs","moe2d"]
cycles = 20
MAXLEN = 120
SIZES = [5,20,50,200]
WINDOWS = [1,3,5,7]

def getproteins():
    files = os.listdir(path)  
    names = set()
    for file in files:
        if '.csv' in file:
            name = file[0:file.rfind('-', 1)]
            names.add(name)
    names = list(names)
    return names

def precess_smiles_as_charlist(data):
    allchars = []
    i = 0
    while i < len(data):
        if data[i] == 'B' and data[i+1] == 'r':
            allchars.append('Br')
            i += 1
        elif data[i] == 'C' and data[i+1] == 'l':
            allchars.append('Cl')
            i += 1
        else:    
            allchars.append(data[i])
        i += 1
    return allchars

def resplit(data_x,data_y):
    smiles = []
    dellist=[]
    for index, smile in enumerate(data_x):
        smile = '~'.join(smile)
        smile = smile.replace('C~l','Cl')
        smile = smile.replace('B~r','Br')
        if 'S~i' in smile or 'e' in smile or "7" in smile or "~B~" in smile:
            dellist.append(index)
        else:
            smile = smile.split('~')
            smiles.append(smile)
    data_y = data_y.drop(dellist)    
    return smiles,data_y

def del_oversize(data_x,data_y,maxlen):
    dellist=[]
    for index,item in enumerate(data_x):
        if len(item) > maxlen:
            dellist.append(index)
    data_x = [x for x in data_x if len(x) <= maxlen]
    dellist = [data_y.index[i] for i in dellist]
    data_y = data_y.drop(dellist)    
    return data_x,data_y

def embedding(SIZE,WINDOW):
    if os.path.exists("embedding_model/mole2vec-gensim-"+str(SIZE)+"-"+str(WINDOW)+".h5"):
        emodel = Word2Vec.load("embedding_model/mole2vec-gensim-"+str(SIZE)+"-"+str(WINDOW)+".h5")
    else:
        smiles_pd = pd.read_csv('../MolecularEmbedding/finalsmiles_pd.csv')
        smileslist = smiles_pd['smiles']
        datawithnextline = '\n'.join(smileslist)
         
        allcharswithn = precess_smiles_as_charlist(datawithnextline)
        
        allsmilesindexlist = []
        smileindexlist = []
        for char in allcharswithn:
            if char != '\n':
                smileindexlist.append(char)
            else:
                allsmilesindexlist.append(smileindexlist)
                smileindexlist = []
            
        emodel = Word2Vec(sentences=allsmilesindexlist, size=SIZE, sg=1, min_count=1, window=WINDOW, seed=42, workers=1)
        emodel.save("embedding_model/mole2vec-gensim-"+str(SIZE)+"-"+str(WINDOW)+".h5")
    return emodel

def e_encode(X,emodel,size,maxlen):
    X0 = []
    for mol in X:
        w = [emodel.wv[i] for i in mol]
        lst = list(chain(*w))
        X0.append(lst)
    for q in X0:
        q.extend((size*maxlen-len(q)) * (0,))
    return X0
    
def model(protein,i):
    if i == 0:
        with open('.\\result\\'+protein+"-"+str(SIZE)+"-"+str(WINDOW)+'.csv','a+') as f:
            f.write('ACC'+','+'SPE'+','+'SEN'+','+'AUC'+'\n')
    data = pd.read_table(path+protein+'-cats.csv',delimiter=',')   
    X = data.iloc[:,0]
    y = data.iloc[:,2]
    X,y = resplit(X,y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    X_train,y_train = del_oversize(X_train,y_train,MAXLEN)
    X_test,y_test = del_oversize(X_test,y_test,MAXLEN)
    emodel = embedding(SIZE,WINDOW)  
    X_train,X_test = e_encode(X_train,emodel,SIZE,MAXLEN),e_encode(X_test,emodel,SIZE,MAXLEN)
    X_train,X_test = np.array(X_train),np.array(X_test)
    
    rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
    answer_rf = rf.predict(X_test)
    
    y_score = rf.predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
    confusion = metrics.confusion_matrix(y_test,answer_rf)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    auc = metrics.auc(fpr,tpr)
    acc = metrics.accuracy_score(y_test,answer_rf)
    sen = TP / (TP+FN)
    spe = TN / (TN+FP)
    print('=========='+protein+"-"+str(SIZE)+"-"+str(WINDOW)+'=========')
    print('The accuracy is: %.3f' %acc)
    print('The specificity is: %.3f' %spe)
    print('The sensitivity is: %.3f' %sen)
    print('The auc is: %.3f' %auc)
    print('============================')
    with open('.\\result\\'+protein+"-"+str(SIZE)+"-"+str(WINDOW)+'.csv','a+') as f:
        f.write('{:.3f}'.format(float(acc))+','+'{:.3f}'.format(float(spe))+','+'{:.3f}'.format(float(sen))+','+'{:.3f}'.format(float(auc))+'\n')

if __name__ == '__main__':
    proteins = getproteins()
    for protein in proteins:
        for SIZE in SIZES:
            for WINDOW in WINDOWS:
                for i in range(cycles):
                    print('=========== '+ str(i) +' ===========')
                    model(protein,i)