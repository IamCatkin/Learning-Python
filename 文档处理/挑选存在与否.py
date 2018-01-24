#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/24 14:44
# @Author  : Catkin
# @File    : 挑选存在与否.py
import pandas as pd

data_a = pd.read_table("UNIPROT_PDB.csv",delimiter=',',keep_default_na=False)
data_b = pd.read_table("Common_uniprot.csv",delimiter=',',keep_default_na=False)
count = 0
for i in data_a.iloc[:,0]:
    if str(i) in list(data_b['Uniprot_ID']):
        data_a.iloc[count,2]='Yes'
    count += 1
data_a.to_csv('results.csv',index=False)