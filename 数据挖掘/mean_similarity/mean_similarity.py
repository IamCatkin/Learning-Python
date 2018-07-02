#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Mon Jun 25 10:02:10 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def getfilenames():
    files = os.listdir()  
    names = set()
    for file in files:
        if 'mean_similarity.csv' in file:
            names.add(file)
    names = list(names)
    return names

def getarray(filename):
    data = pd.read_csv(filename)
    data1 = np.array(data)
    data2 = np.triu(data1,1)

# option 1
#    myarray = np.array(0)
#    for i in range(len(data2)-1):
#        myarray = np.append(myarray,data2[i,i+1:])
#    myarray = myarray[1:]

# option 2
    mylist = []
    for i in range(len(data2)-1):
        mylist.append(data2[i,i+1:])
    mylist1 = [num for elem in mylist for num in elem]
    myarray = np.array(mylist1)

    return myarray

def scatter(x,y,i,j,k):
    plt.figure(figsize=(16, 9))
    plt.scatter(x, y)
#    plt.show()
    plt.savefig(j+'_'+k+'_'+str(i+1)+'.pdf')
    plt.close()

def gettri():
    result = pd.DataFrame()
    for filename in filenames:
        array = getarray(filename)
        dataframe = pd.Series(array)
        result[filename[7:-20]] = dataframe
    result.to_csv('result.csv',index=False)
    return result

def getpearson(loops):
    pearsons = pd.DataFrame()
    maccs_rdkit = []
    maccs_atompair = []
    maccs_morgan = []
    rdkit_atompair = []
    rdkit_morgan = []
    atompair_morgan = []
    for i in range(loops):
        index = np.random.randint(result.shape[0],size=1000)
        sdata = result.iloc[index,:]
        pearson = sdata.corr()
        maccs_rdkit.append(pearson.iloc[0,1])
        maccs_atompair.append(pearson.iloc[0,2])
        maccs_morgan.append(pearson.iloc[0,3])
        rdkit_atompair.append(pearson.iloc[1,2])
        rdkit_morgan.append(pearson.iloc[1,3])
        atompair_morgan.append(pearson.iloc[2,3])
        x = sdata['maccs']
        y = sdata['rdkit']
        scatter(x,y,i,'maccs','rdkit')
        x = sdata['maccs']
        y = sdata['atompair']
        scatter(x,y,i,'maccs','atompair')
        x = sdata['maccs']
        y = sdata['morgan']
        scatter(x,y,i,'maccs','morgan')
        x = sdata['rdkit']
        y = sdata['atompair']
        scatter(x,y,i,'rdkit','atompair')
        x = sdata['rdkit']
        y = sdata['morgan']
        scatter(x,y,i,'rdkit','morgan')
        x = sdata['atompair']
        y = sdata['morgan']
        scatter(x,y,i,'atompair','morgan')
    pearsons['maccs_rdkit'] = maccs_rdkit
    pearsons['maccs_atompair'] = maccs_atompair
    pearsons['maccs_morgan'] = maccs_morgan
    pearsons['rdkit_atompair'] = rdkit_atompair
    pearsons['rdkit_morgan'] = rdkit_morgan
    pearsons['atompair_morgan'] = atompair_morgan
    pearsons.to_csv('pearson'+'_'+str(loops)+'.csv',index=False)
    return pearsons
    
if __name__ == '__main__':
    filenames = getfilenames()
    result = gettri()
    loops = 100
    pearsons = getpearson(loops)
    
#    x = result['maccs'][index]
#    y = result['rdkit'][index]
#    scatter(x,y)
    
#    p = result.corr()
