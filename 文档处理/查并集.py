#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/22 14:30
# @Author  : Catkin
# @File    : 查并集.py
import pandas as pd

data = pd.read_table("Uniprot_PDB.csv",delimiter=',',keep_default_na=False)

pdb = data.iloc[:,1]
duplicate = pdb.duplicated()
dup_index = duplicate[duplicate==True]
dup_list=[]
for i in dup_index.index:
    dup_list.append(data.iloc[i,1])
for j in range(0,len(data.index)):
    if data.iloc[j,1] in dup_list:
        with open("dup_result.csv","a+") as f:
            f.write(data.iloc[j,1]+','+data.iloc[j,0]+'\n')
            f.close()