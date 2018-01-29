#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/29 10:10
# @Author  : Catkin
# @File    : errorbar.py
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(37, 15)


data = pd.read_table("results_20_2.csv",delimiter=',')
sort_data = data.sort_values(["moe2d_auc_mean"],ascending=False)
x = [i for i in range(1,len(sort_data.index)+1)]
y = sort_data.iloc[:,23]
std = [j**0.5 for j in sort_data.iloc[:,24]]
plt.errorbar(x, y, yerr=std)
plt.title('The errorbar of moe2d_auc')
plt.ylabel('AUC')
plt.xlabel('Targets')
plt.show()
fig.savefig('moe2d_auc.svg', dpi=1200)