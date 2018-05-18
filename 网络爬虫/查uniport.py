#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Fri May 18 12:32:10 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import re
import string
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

data = pd.read_excel('Target_3219_uniprot.xls')
origin = 'http://www.uniprot.org/uniprot/'
uids = []
kbs = []
proteins = []
genes = []
organisms = []
familys = []
try:
    with tqdm(total=len(data['Uniprot_ID'])) as pbar:  
        for uid in data['Uniprot_ID']:
            url = origin + uid
            page = requests.get(url).text
            bsObj = BeautifulSoup(page, 'lxml')
            kb0 = bsObj.find('span',property="schema:alternateName")
            if kb0 is None:
                kb = None
            else:
                kb = kb0.get_text().strip(string.punctuation)
            protein0 = bsObj.find('h1',property="schema:name")
            if protein0 is None:
                protein = None
            else:
                protein = protein0.get_text()
            gene0 = bsObj.find('div',id="content-gene")
            if gene0 is None:
                gene = None
            else:
                gene = gene0.get_text()
            organism0 = bsObj.find('div',id="content-organism")
            if organism0 is None:
                organism = None
            else:
                organism = organism0.get_text()
            patterns = re.compile('<div class="annotation" property="schema:hasPart" typeof="schema:CreativeWork">Belongs to the <a href=.*?>(.*?)</a>')
            family0 =  re.findall(patterns,page)
            if family0:
                family = family0[0]
            else:
                family = None
            uids.append(uid)
            kbs.append(kb)  
            proteins.append(protein)
            genes.append(gene)
            organisms.append(organism)
            familys.append(family)
            pbar.update(1)
finally:
    data1 = pd.DataFrame()
    data1['Uniprot_ID'] = uids
    data1['UniProtKB'] = kbs
    data1['Protein'] = proteins
    data1['Gene'] = genes
    data1['Organism'] = organisms
    data1['Family'] = familys
    data1.to_csv('done.csv',index=False)
    