#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Fri Jul  6 11:28:24 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('result.csv')

fig = plt.figure(figsize=(16, 9))
sns.violinplot(data=data)
fig.savefig('violin.pdf')