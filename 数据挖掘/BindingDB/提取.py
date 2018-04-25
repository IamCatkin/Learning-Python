#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/4/23 19:40
# @Author  : Catkin
# @Site    : blog.catkin.moe
# @File    : 提取.py
import pandas as  pd

def getsmi():
    data = pd.read_table('bingdingdb_2_2_4.csv',sep=',')
    s = set()
    for i in data['smi']:
        s.add(i)
    df = pd.DataFrame()
    df['smi'] = list(s)
    df.to_csv('smi.csv',index=True)
    
def checknan():
    df = pd.read_table('smi_moe.csv',sep=',')
    na = df[df.isnull().values==True]
    na = na.drop_duplicates()
    return na

def getuniport():
    data = pd.read_table('bingdingdb_2_2_4.csv',sep=',')
    seq = data['BindingDB Target Chain  Sequence']
    uni = data['UniProt (SwissProt) Primary ID of Target Chain']
    dic = dict(zip(uni, seq))
    result = pd.DataFrame()
    result['uni'] = dic.keys()
    result['seq'] = dic.values()
    result.to_csv('uni.csv', index=True)

def toindex():
    data = pd.read_table('bingdingdb_2_2_4.csv',sep=',')
    result = data[['smi','UniProt (SwissProt) Primary ID of Target Chain','Ki (nM)']]
    smi = pd.read_table('smi.csv',sep=',')
    uni = pd.read_table('uni.csv',sep=',')
    s_dic = dict(zip(smi['smi'],smi['index']))
    u_dic = dict(zip(uni['uni'],uni['index']))
    result1 = result.copy()
    for i in range(len(result)):
        s = s_dic[result.iloc[i,0]]
        result1.iloc[i,0] = s
        u = u_dic[result.iloc[i,1]]
        result1.iloc[i,1] = u
    result1.to_csv('s-u.csv', index=False)
    
if __name__ == '__main__':
    pass