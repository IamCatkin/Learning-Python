#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Apr 19 18:12:41 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as  pd

d=dict()
data = pd.read_table('done_wash_wash_de2_inchi_2.csv',sep=',')
ki = data['Ki (nM)']
dic = data['dict']
for i in dic:
    x = ki[dic[dic==i].index].values
    x = x.astype(str)
    x = list(x)
    x1=';'.join(x)
    d[i] = x1
df = pd.DataFrame()

df['key'] = d.keys()
df['value'] = d.values()
df.to_csv('done.csv',index=False,header=False)