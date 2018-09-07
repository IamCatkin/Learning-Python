#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Mon Aug 27 16:36:19 2018
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
    fig = plt.figure(figsize=(16, 9))
    plt.scatter(x, y)
#    plt.show()
    fig.savefig(j+'_'+k+'_'+str(i+1)+'.pdf')
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
    BioTP_atompair = []
    BioTP_maccs = []
    BioTP_rdkit = []
    BioTP_morgan = []
    for i in range(loops):
        index = np.random.randint(result.shape[0],size=1000)
        sdata = result.iloc[index,:]
        pearson = sdata.corr()
        BioTP_atompair.append(pearson.iloc[4,0])
        BioTP_maccs.append(pearson.iloc[4,1])
        BioTP_rdkit.append(pearson.iloc[4,2])
        BioTP_morgan.append(pearson.iloc[4,3])
        x = sdata['BioTP']
        y = sdata['atompair']
        scatter(x,y,i,'BioTP','atompair')
        x = sdata['BioTP']
        y = sdata['maccs']
        scatter(x,y,i,'BioTP','maccs')
        x = sdata['BioTP']
        y = sdata['rdkit']
        scatter(x,y,i,'BioTP','rdkit')
        x = sdata['BioTP']
        y = sdata['morgan']
        scatter(x,y,i,'BioTP','morgan')
    pearsons['BioTP_atompair'] = BioTP_atompair
    pearsons['BioTP_maccs'] = BioTP_maccs
    pearsons['BioTP_rdkit'] = BioTP_rdkit
    pearsons['BioTP_morgan'] = BioTP_morgan
    pearsons.to_csv('pearson'+'_'+str(loops)+'.csv',index=False)
    return pearsons

if __name__ == '__main__':
    filenames = getfilenames()
    result = gettri()
    pear = True
    if pear is True:
        loops = 100
        pearsons = getpearson(loops)