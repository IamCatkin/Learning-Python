#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Apr 26 15:52:49 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as pd
import matplotlib.pyplot as plt

#绘制直方图
def drawHist(data):
    plt.hist(data,28)
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('log_dropped_s-u.csv')
    log_ki = data['log_Ki']
    ln_ki = data['ln_Ki']
    drawHist(log_ki)
    drawHist(ln_ki)
