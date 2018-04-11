#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 11:24
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : Inchlkey.py
import os
import pandas as pd

path = '.\\22 txt\\'

def getfilenames(files):
    filenames = []
    for file in files:
        if '.txt' in file:
            name = file[:-10]
            filenames.append(name)
    return filenames

def search(name):
    data = pd.read_table(path+name,delimiter='\t')  
    total[name] = None
    count = 0
    for i in total['Inchlkey']:
        if i in list(data.iloc[:,3]):
            total.iloc[count,-1] = 1
        else:
            total.iloc[count,-1] = 0
        count += 1

if __name__ == '__main__':
    files = os.listdir(path)
    total = pd.read_table('modeling total data.txt',delimiter='\t')  
    for name in files:
        search(name)
    total.to_csv('new_total.csv',index=False)
