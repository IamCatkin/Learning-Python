#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Wed Aug 22 14:43:03 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import re
import pandas as pd

data = pd.read_excel('ddi.xlsx')
ddidata = data.iloc[2:,0]
dbs = []
patterns = 'DB\d{5}'
for ddi in ddidata:
    db =  list(set(re.findall(patterns,ddi)))
    db = sorted(db)
    dbs.append(db)
alldb = pd.DataFrame(dbs)
dup = alldb.duplicated(keep=False)
dups = alldb[dup]
index = dups.index + 2
result = data.iloc[index,:]
result.to_csv('result.csv',index=False,header=False)
