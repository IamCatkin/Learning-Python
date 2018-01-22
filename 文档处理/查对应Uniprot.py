#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/22 10:30
# @Author  : Catkin
# @File    : 查对应Uniprot(字典方式).py
import pandas as pd

data_a = pd.read_table("35170.csv",delimiter=',',keep_default_na=False)
data_b = pd.read_table("Uniprot_PDB.csv",delimiter=',',keep_default_na=False)
results = pd.DataFrame()
results["uniport"]='nan'
results["pdb"]="nan"
data_a["uniport"]='nan'

pdb = data_b.iloc[:,1]
uni = data_b.iloc[:,0]
dic = dict(zip(pdb, uni))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,3] != "":
        data_a.iloc[i,6]=dic[data_a.iloc[i,3]]
data_a.to_csv('trans_result.csv',index=True)

################################################
#       弃用版本，效率太低
#import pandas as pd
#data_a = pd.read_table("35170.csv",delimiter=',',keep_default_na=False)
#data_b = pd.read_table("Uniprot_PDB.csv",delimiter=',',keep_default_na=False)
#results = pd.DataFrame()
#results["uniport"]='nan'
#results["pdb"]="nan"
#data_a["uniport"]='nan'
#for i in range(0,len(data_a.index)):
#    for j in range(0,len(data_b.index)):
#        if data_a.iloc[i,3]==data_b.iloc[j,1]:
#            data_a.iloc[i,6]=data_b.iloc[j,0]
#data_a.to_csv('trans_result.csv',index=True)
#
#for k in range(0,len(data_b.index)):
#    count = 0
#    if data_b.iloc[k,1] not in data_a.iloc[:,3]:
#        results.iloc[count,0]=data_b.iloc[k,0]
#        results.iloc[count,1]=data_b.iloc[k,1]
#        count += 1
#results.to_csv('search_result.csv',index=True)
##################################################