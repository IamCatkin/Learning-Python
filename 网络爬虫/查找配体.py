#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/12 15:47
# @Author  : Catkin
# @File    : 查找配体.py
import requests
import socket
from urllib.request import urlretrieve
import re
from openpyxl import load_workbook
from time import sleep

wb = load_workbook('Uniprot_PDB.xlsx')
sheet = wb['Sheet1']
try:
    for i in range(2,18113):
        uniprot = sheet.cell(row=i,column=1).value
        pdb = sheet.cell(row=i,column=2).value
        urls = "http://www.rcsb.org/pdb/explore/explore.do?structureId="+str(pdb)
        page = requests.get(url=urls,timeout=50.0).text
        if "ligandPage" in page:
            patterns = re.compile('https://files.rcsb.org/ligands/download/(.*?).cif')
            liagands = re.findall(patterns,page)
            if len(liagands) != 0:
                for j in range(len(liagands)):
                    print(liagands[j])
                    filename = ".\\ligand\\" + uniprot + '_' + str(pdb) + '_' + liagands[j] + '.sdf'
                    downloadlinks = "http://www.rcsb.org/pdb/download/downloadLigandFiles.do?ligandIdList=" + liagands[j] + "&structIdList=" + str(pdb) + "&instanceType=all&excludeUnobserved=false&includeHydrogens=false"
                    socket.setdefaulttimeout(10.0)
                    urlretrieve(downloadlinks, filename)
                    sheet.cell(row=i,column=4+j).value=liagands[j]
            else:
                sheet.cell(row=i, column=4).value = 'Fault'
                print('Fault')
        elif "No results were found" in page:
            with open('error.txt','a+') as f:
                print('Error')
                f.write(str(i)+"\n")
                f.close()
        elif "ligandPage" not in page:
            sheet.cell(row=i,column=4).value='None'
            print('None')
        sleep(0.6)
finally:
    wb.save('results.xlsx')

