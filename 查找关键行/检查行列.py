#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2017/10/31 10:38
# @Author  : Catkin
# @File    : 检查行列.py
import xlrd
import xlwt

data = xlrd.open_workbook(r'text.xlsx')
rtable = data.sheets()[0]
wbook = xlwt.Workbook(encoding = 'utf-8',style_compression = 0)
wtable = wbook.add_sheet('sheet1',cell_overwrite_ok = True)

print(rtable.nrows)
print(rtable.ncols)
