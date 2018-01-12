#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2017/10/31 9:42
# @Author  : Catkin
# @File    : 提取关键词行.py
import xlrd
import xlwt

data = xlrd.open_workbook(r'text.xlsx')
rtable = data.sheets()[0]
wbook = xlwt.Workbook(encoding = 'utf-8',style_compression = 0)
wtable = wbook.add_sheet('sheet1',cell_overwrite_ok = True)

count = 0
keyword = format('hepatic failure').lower()
for i in range(0, 8224):
    if keyword in rtable.cell(i,3).value.lower():
        for j in range(0,4):
            wtable.write(i, j, rtable.row_values(i)[j])
            count += 1
print(count)
wbook.save(r'result-1.xls')
