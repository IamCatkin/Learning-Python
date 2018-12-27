#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 10:14
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#绘制直方图
def drawHist(data):
    plt.hist(data,28)
    plt.show()

def seeDistribution(data):
    dt = pd.DataFrame()
    median = data.median()
    e = []
    p = []
    n = []
    for i in range(4,9):
        positive = len(data['log_Ki'][data['log_Ki']<=-i])
        negative = len(data['log_Ki'][data['log_Ki']>-i])
        e.append(i)
        p.append(positive)
        n.append(negative)
    dt["exp"] = e
    dt["positive"] = p
    dt["negative"] = n
    return median,dt

def countFramework():
    frame = pd.read_csv('smi_framework.csv')
    frame = frame.fillna('None')
    murcko_counts = frame['Murcko'].value_counts()
    carbon_counts = frame['Carbon'].value_counts()
    murcko_counts.to_csv('murcko_counts.csv')
    carbon_counts.to_csv('carbon_counts.csv')

def analyseLip(lip):
    plt.hist(lip['lip_acc'],14)
    plt.show()
    plt.hist(lip['lip_don'],7)
    plt.show()
    plt.hist(lip['b_rotN'],35)
    plt.show()
    plt.hist(lip['logP(o/w)'],13)
    plt.show()
    plt.hist(lip['Weight'],14)
    plt.show()

def analyseProtein(pro):
    counts = {}
    strings = ''.join(list(pro['seq'].values.flatten()))
    for j in strings:
        counts[j] = counts.get(j, 0) + 1
    return counts

if __name__ == '__main__':
    data = pd.read_csv('log_dropped_s-u_1.csv')
    lip = pd.read_csv('smi_de.csv')
    pro = pd.read_csv('uni.csv')
#    log_ki = data['log_Ki']
#    drawHist(log_ki)
    median,dt = seeDistribution(data) 
    countFramework()
    analyseLip(lip)
    counts = analyseProtein(pro)