#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 23:01
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : boxplot.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

#########     参数     ###########
path = '.\\result\\'
descriptors = ["cats","ecfp4","maccs","moe2d"]
targets = ['ACC','SPE','SEN','AUC']

def getproteins():
    files = os.listdir(path)  
    names = set()
    for file in files:
        if '.csv' in file:
            name = file[0:file.rfind('-', 1)]
            names.add(name)
    names = list(names)
    return names

def getboxplot_RF():
    for target in targets:
        df = pd.DataFrame()
        for descriptor in descriptors:
            for protein in proteins:
                data = pd.read_table(path+protein+'-'+descriptor+'.csv',delimiter=',')   
                df[protein]=data[target]
            plt.figure(figsize=(16, 9))
            sns.set_style("white")
            sns.boxplot(data=df)
            plt.savefig('.\\boxplot\\'+descriptor+'-'+target+'.pdf')
            plt.close()

def getboxplot_MLP():
    for target in targets:
        df = pd.DataFrame()
        for protein in proteins:
            data = pd.read_table(path+protein+'-MLP.csv',delimiter=',')   
            df[protein]=data[target]
        plt.figure(figsize=(16, 9))
        sns.set_style("white")
        sns.boxplot(data=df)
        plt.savefig('.\\boxplot\\'+'MLP'+'-'+target+'.pdf')
        plt.close()

def getboxplot_CNN():
    for target in targets:
        df = pd.DataFrame()
        for protein in proteins:
            data = pd.read_table(path+protein+'-CNN.csv',delimiter=',')   
            df[protein]=data[target]
        plt.figure(figsize=(16, 9))
        sns.set_style("white")
        sns.boxplot(data=df)
        plt.savefig('.\\boxplot\\'+'CNN'+'-'+target+'.pdf')
        plt.close()

if __name__ == '__main__':
    proteins = getproteins()
#    getboxplot_RF()
#    getboxplot_MLP()
#    getboxplot_CNN()