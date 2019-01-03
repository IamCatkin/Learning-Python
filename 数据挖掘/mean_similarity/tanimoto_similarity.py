#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/01/02 20:10
# @Author  : Catkin
# @Website : blog.catkin.moe
import time
import numpy as np
import pandas as pd

#parameter
PRECISION = 3
PATH = './17_tragets_cats.txt'

def loadData(path):
    data = pd.read_csv(path)
    return data.iloc[:,6:]

def getTanimotoCoefficient(p1,p2):
    x1 = p1[p1>0]
    x2 = p2[p2>0]
    si = {}
    for item in list(x1.index):
        if item in list(x2.index):
            si[item] = 1
    n = len(si)
    if n == 0:
        return 0
    sum1 = sum([pow(it,2) for it in x1])
    sum2 = sum([pow(it,2) for it in x2])
    sumco = sum([x1[it]*x2[it] for it in si])
    return sumco/(sum1+sum2-sumco)

def getMatrix(data):
    K = len(data)
    correl = np.empty((K, K), dtype=float)
    for i, ac in data.iterrows():
        for j, bc in data.iterrows():
            if i > j:
                continue
            elif i == j:
                c = 1.
            else:
                c = getTanimotoCoefficient(ac, bc)
            correl[i, j] = c
            correl[j, i] = c
    return correl

def main():
    start = time.time()
    data = loadData(PATH)
    corr = getMatrix(data)
    end = time.time()
    print("{} seconds used.".format(round(end-start)))
    corr = np.around(corr,decimals=PRECISION)
    corr = pd.DataFrame(corr)
    corr.to_csv('tanimoto.csv',sep=',',header=None,index=None)
            
if __name__ == '__main__':
    main()