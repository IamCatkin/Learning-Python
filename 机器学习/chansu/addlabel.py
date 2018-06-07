#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Jun  7 10:24:19 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import os
import pandas as pd

#########   parameters  ########
descriptors = ["cats","maccs","moe2d"]

def getfilenames():
    path = 'cats'
    files = os.listdir(path)
    names = set()
    for file in files:
        if '.txt' in file:
            name = file[:-4]
            names.add(name)
    names = list(names)
    return names

def addlabel(filename):
    for descriptor in descriptors:
        path = ".\\"+descriptor+"\\"
        data = pd.read_table(path+filename+".txt",sep='\t')
        data['label'] = None
        value = data['value']
        median = value.median()
        for i in range(len(data.index)):
            if data.iloc[i,3] < median:
                data.iloc[i,len(data.columns)-1] = 1
            else:
                data.iloc[i,len(data.columns)-1] = 0
        data.to_csv(path+filename+'.csv',sep='\t',index=False)

def analysis():
    path = ".\\cats\\"
    with open('analysis.csv','a+') as f:
        f.write('filename'+','+'counts'+','+'median'+'\n')
    for filename in filenames:
        data = pd.read_table(path+filename+".txt",sep='\t')
        counts = len(data)
        value = data['value']
        median = value.median()
        with open('analysis.csv','a+') as f:
            f.write(filename+','+str(counts)+','+str(median)+'\n')
          
if __name__ == '__main__':
    filenames = getfilenames()
    for filename in filenames:
        addlabel(filename)
    analysis()