#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/12 15:47
# @Author  : Catkin
# @File    : 查找配体.py
import requests
import re
from openpyxl import load_workbook
from time import sleep

wb = load_workbook('4.xlsx')
sheet = wb['Sheet1']
for i in range(2,6):
    uniprot = sheet.cell(row=i,column=1).value
    pdb = sheet.cell(row=i,column=2).value
    urls = "http://www.rcsb.org/pdb/explore/explore.do?structureId="+pdb
    page = requests.get(url=urls,timeout=5).text
    if "ligandPage" in page:
        patterns = re.compile('https://files.rcsb.org/ligands/download/(.*?).cif')
        liagands = re.findall(patterns,page)
        for j in range(len(liagands)):
            print(liagands[j])
            sheet.cell(row=i,column=3+j).value=liagands[j]
    else:
        sheet.cell(row=i,column=3).value='None'
        print('None')
wb.save('44.xlsx')