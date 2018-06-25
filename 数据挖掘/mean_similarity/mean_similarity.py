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
        if 'mean_similarity' in file:
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

def scatter(x,y):
    
    plt.figure(figsize=(16, 9))
    plt.scatter(x, y)
    plt.show()

if __name__ == '__main__':
    filenames = getfilenames()
    result = pd.DataFrame()
    for filename in filenames:
        array = getarray(filename)
        dataframe = pd.Series(array)
        result[filename[7:-20]] = dataframe
    
    result.to_csv('result.csv',index=False)
    
    index = np.random.randint(result.shape[0],size=10000)
    x = result['maccs'][index]
    y = result['rdkit'][index]
    scatter(x,y)
    
    p = result.corr()
