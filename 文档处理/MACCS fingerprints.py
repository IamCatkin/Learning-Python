#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 09:02
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : MACCS fingerprints.py
import pandas as pd
import os

def getnames():
    files = os.listdir()  
    names = set()
    for file in files:
        if '.txt' in file:
            name = file[:-4]
            names.add(name)
    names = list(names)
    return names    

def trans(name):
    data = pd.read_table(name+'.txt',sep='\t')
    head = data.iloc[:,:6]
    maccs = data.iloc[:,6:]
    maccs[maccs>1] = 1
    maccs = pd.concat([head,maccs],axis=1)
    maccs.to_csv(name+'.csv',index=False)

if __name__ == '__main__':
    names = getnames()
    for name in names:
        trans(name)