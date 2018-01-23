#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/23 9:30
# @Author  : Catkin
# @File    : 添加信息.py
import pandas as pd

data_a = pd.read_table("14736.csv",delimiter=',',keep_default_na=False)
data_b = pd.read_table("General_PL_Data.csv",delimiter=',',keep_default_na=False)
data_c = pd.read_table("General_PL_Name.csv",delimiter=',',keep_default_na=False)

#############         添加uniport       ###################
pdb = data_c.iloc[:,0]
uni = data_c.iloc[:,2]
dic = dict(zip(pdb, uni))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,7]='Error'
    else:
        data_a.iloc[i,7]=dic[data_a.iloc[i,0].lower()]
###############      添加year         ######################
year = data_c.iloc[:,1]
dic = dict(zip(pdb, year))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,13]='Error'
    else:
        data_a.iloc[i,13]=dic[data_a.iloc[i,0].lower()]
#############         添加target_name           #################
target = data_c.iloc[:,3]
dic = dict(zip(pdb, target))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,8]='Error'
    else:
        data_a.iloc[i,8]=dic[data_a.iloc[i,0].lower()]
##############           添加code         #################3
pdb = data_b.iloc[:,0]
code = data_b.iloc[:,3]
dic = dict(zip(pdb, code))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,10]='Error'
    else:
        data_a.iloc[i,10]=dic[data_a.iloc[i,0].lower()]
#############           添加resolution         #################
pdb = data_b.iloc[:,0]
code = data_b.iloc[:,1]
dic = dict(zip(pdb, code))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,9]='Error'
    else:
        data_a.iloc[i,9]=dic[data_a.iloc[i,0].lower()]
#############           添加value         #################
pdb = data_b.iloc[:,0]
value = data_b.iloc[:,4]
dic = dict(zip(pdb, value))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,11]='Error'
    else:
        data_a.iloc[i,11]=dic[data_a.iloc[i,0].lower()]
##############           添加ligand         #################
pdb = data_b.iloc[:,0]
ligand = data_b.iloc[:,6]
dic = dict(zip(pdb, ligand))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,12]='Error'
    else:
        data_a.iloc[i,12]=dic[data_a.iloc[i,0].lower()]
##############           添加ref         #################
pdb = data_b.iloc[:,0]
ref = data_b.iloc[:,5]
dic = dict(zip(pdb, ref))
for i in range(0,len(data_a.index)):
    if data_a.iloc[i,0] in ['4jan']:
        data_a.iloc[i,14]='Error'
    else:
        data_a.iloc[i,14]=dic[data_a.iloc[i,0].lower()]

data_a.to_csv('trans_14736.csv',index=False)
