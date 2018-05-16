#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Wed May 16 11:01:08 2018
# @Author  : Catkin
# @Website : blog.catkin.moe

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

#########     参数     ###########
path = '.\\result\\'
descriptors = ["cats","ecfp4","maccs","moe2d"]
targets = ['ACC','SPE','SEN','AUC']
proteins = ['PI3K','Jak3','mTOR','VEGFR2','HER2','JAK2','MEK1','IDO1','AXL','AKT1','RSK2','Bcl2']
SIZES = [5,20,50,200]
WINDOWS = [1,3,5,7]

def getblox():
    for protein in proteins:
        for target in targets:
            df = pd.DataFrame()
            for SIZE in SIZES:
                for WINDOW in WINDOWS:
                    para = str(SIZE)+"-"+str(WINDOW)
                    data = pd.read_table(path+protein+"-"+str(SIZE)+"-"+str(WINDOW)+'.csv',delimiter=',')   
                    df[para] = data[target]
            plt.figure(figsize=(16, 9))
            sns.set_style("white")
            sns.boxplot(data=df)
            plt.savefig('.\\select\\'+'Embedding-RF'+"-"+protein+'-'+target+'.pdf')
            plt.close()

def getallblox():
    for target in targets:
        alldf = pd.DataFrame()
        for protein in proteins:
            df = pd.DataFrame()
            for SIZE in SIZES:
                for WINDOW in WINDOWS:
                    para = str(SIZE)+"-"+str(WINDOW)
                    data = pd.read_table(path+protein+"-"+str(SIZE)+"-"+str(WINDOW)+'.csv',delimiter=',')   
                    df[para] = data[target]
            alldf = pd.concat([alldf,df])
        plt.figure(figsize=(16, 9))
        sns.set_style("white")
        sns.boxplot(data=alldf)
        plt.savefig('.\\select\\'+'Embedding-RF'+'-'+target+'.pdf')
        plt.close()
                    
def gettable():
    result = pd.DataFrame()
    for protein in proteins:
        for target in targets:
            df = pd.DataFrame()
            for SIZE in SIZES:
                for WINDOW in WINDOWS:
                    para = str(SIZE)+"_"+str(WINDOW)
                    data = pd.read_table(path+protein+"-"+str(SIZE)+"-"+str(WINDOW)+'.csv',delimiter=',')   
                    df[para] = data[target]
            p = df.describe().iloc[1:3,:]
            p.index = [protein+"-"+target+"-mean",protein+"-"+target+"-std"]
            result = pd.concat([result,p])
    result.to_csv('.\\select\\'+'Embedding-RF.csv')
    return result
                        
if __name__ == '__main__':
#    getblox()
#    result = gettable()
    getallblox()
            