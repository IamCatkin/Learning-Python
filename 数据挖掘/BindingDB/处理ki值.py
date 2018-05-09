#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Apr 26 12:29:38 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as pd
import math

def tofloat():
    ki = data['Ki']
    f_ki = []
    for i in ki:
        i = i.split(';')
        i = [float(f) for f in i]
        f_ki.append(i)
    f_ki = pd.Series(f_ki)
    return f_ki

def getki():
    n_ki = []
    for i in f_ki:
        if len(i) == 1:
            p = i[0]
        elif len(i) == 2:
            if max(i) / min(i) > 10:
                p = None
            else:
                p = sum(i) / len(i)
        else:
            i = pd.Series(i)
            p = i.median()
        n_ki.append(p)
    n_ki = pd.Series(n_ki)
    data['new_Ki(nM)'] = n_ki
    data.to_csv('new_s-u.csv',index=False)

def getlog():
    data = pd.read_table('dropped_s-u.csv',sep=',')
    n_ki = data['new_Ki(nM)']
    log_ki = []
    for i in n_ki:
        p = math.log10(i)
        p = p - 9
        log_ki.append(p)
    log_ki = pd.Series(log_ki)
    ln_ki = []
    for i in n_ki:
        p = math.log(i)
        p = p + math.log(10**-9)
        ln_ki.append(p)
    log_ki = pd.Series(log_ki)
    ln_ki = pd.Series(ln_ki)
    format = lambda x: '%.3f' % x
    log_ki = log_ki.apply(format)
    ln_ki = ln_ki.apply(format)
    data['log_Ki'] = log_ki
    data['ln_Ki'] = ln_ki
    data.to_csv('log_dropped_s-u.csv',index=False)
    return data

def checknan():
    df = pd.read_table('new_s-u.csv',sep=',')
    na = df[df.isnull().values==True]
    na = na.drop_duplicates()
    df_b = df.dropna(axis=0)
    df_b.to_csv('dropped_s-u.csv',index=False)
    return na

if __name__ == '__main__':
    data = pd.read_csv('s-u.csv')
    f_ki = tofloat()
    getki()
    na = checknan()
    getlog()