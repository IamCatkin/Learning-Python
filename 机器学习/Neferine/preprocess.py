#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Sep 20 16:48:17 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

def dataloader(d):
    data = pd.read_excel(d+'.xlsx',['cats','maccs','moe2d'])
    return data

def preprocess(descriptor,data):
    p = data[descriptor]
    names = p.columns
    values = []
    targets = []
    descriptors = []
    for name in names:
        values.extend(p[name])
        targets.extend((name,)*len(p))
    descriptors.extend((descriptor,)*len(values))
    return values,targets,descriptors

def boxplot(result,d):
    sns.set(style="ticks", palette="pastel")
    fig = plt.figure(figsize=(16, 12))
    sns.boxplot(x="targets", y="values",
            hue="descriptors", palette=["m", "g", 'r'],
            data=result,linewidth=2)
    plt.legend(loc="lower right",fontsize=16)
    fig.savefig(d+'.pdf')

def getresult(data):
    v = []
    t = []
    ds = []    
    for descriptor in ['cats','maccs','moe2d']:
        a,b,c = preprocess(descriptor,data)
        v.extend(a)
        t.extend(b)
        ds.extend(c)
    result = pd.DataFrame()
    result['values'] = v
    result['descriptors'] = ds
    result['targets'] = t
    return result

def main():
    for d in ['acc','sen','spe','AUC']:
        data = dataloader(d)
        result = getresult(data)
        boxplot(result,d)
    
if __name__ == '__main__':
    main()
