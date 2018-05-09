#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Apr 26 10:08:03 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as pd

def smi_count():
    data = pd.read_csv('s-u.csv')
    smi = pd.read_csv('smi.csv')['index']
    T = []
    counts = []
    for i in smi:
        x = data[data['smi']==i]['UniProt (SwissProt) Primary ID of Target Chain'].values
        x = list(x)
        count = len(x)
        x1 = ';'.join(x)
        T.append(x1)
        counts.append(count)
    df = pd.DataFrame()
    df['compound'] = smi
    df['target'] = T
    df['count'] = counts
    df.to_csv('smi_c.csv',index=False)

def uni_count():
    data = pd.read_csv('s-u.csv')
    uni = pd.read_csv('uni.csv')['index']
    C = []
    counts = []
    for i in uni:
        x = data[data['UniProt (SwissProt) Primary ID of Target Chain']==i]['smi'].values
        x = list(x)
        count = len(x)
        x1 = ';'.join(x)
        C.append(x1)
        counts.append(count)
    df = pd.DataFrame()
    df['target'] = uni
    df['compound'] = C
    df['count'] = counts
    df.to_csv('uni_c.csv',index=False)

if __name__ == '__main__':
    smi_count()
    uni_count()