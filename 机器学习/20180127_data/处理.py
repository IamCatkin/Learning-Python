#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 11:24
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : 处理.py
import os
import pandas as pd

def getproteins():
    files = os.listdir()  
    names = set()
    for file in files:
        if '.csv' in file:
            name = file[0:file.rfind('-', 1)]
            names.add(name)
    names = list(names)
    return names

def getvalues(protein):
    data_v = pd.read_table(protein + '-CNN_val.csv',delimiter=',')   
    data_t = pd.read_table(protein + '-CNN_test.csv',delimiter=',') 
    table_a = data_v['ACC']
    result = []
    for i in range(20):
        locals()['series'+str(i)] = table_a[table_a.index % 20 == i]
        mean = locals()['series'+str(i)].mean()
        result.append(mean)
    index = result.index(max(result))
    data_t_c = data_t[data_t.index % 20 == index].iloc[:,1:]
    data_t_c.to_csv(protein + '-CNN.csv',index=False)

if __name__ == '__main__':
    proteins = getproteins()
    for protein in proteins:
        getvalues(protein)