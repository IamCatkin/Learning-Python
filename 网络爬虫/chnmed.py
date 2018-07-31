#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Tue Jul 31 16:37:53 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import time
from tqdm import tqdm
import pandas as pd
from lxml import html
from selenium import webdriver

def dataloader():
    data = pd.read_excel('database.xlsx')
    cas = data['CAS']
    return data,cas

def jscrawler(c):
    url = 'https://www.ncbi.nlm.nih.gov/pccompound?term=' + c    
    fireFoxOptions = webdriver.FirefoxOptions()
    fireFoxOptions.set_headless()
    brower = webdriver.Firefox(firefox_options=fireFoxOptions)
    brower.get(url)
    time.sleep(10)
    page = brower.page_source
    brower.close()
    tree = html.fromstring(page)
    cid_0 = tree.xpath('//*[@id="summary-app"]/div[1]/div[3]/div[1]/table/tbody/tr[1]/td')
    if cid_0:
        cid = cid_0[0].text
    else:
        cid = 'None'
    smile_0 = tree.xpath('//*[@id="Canonical-SMILES"]/div[2]/div[1]')
    if smile_0:
        smile = smile_0[0].text
    else:
        smile = 'None'
    name_0 = tree.xpath('//*[@id="summary-app"]/div[1]/div[2]/div/h1/span')
    if name_0:
        name = name_0[0].text
    else:
        name = 'None'
    return cid,smile,name
        
def datacheker(cas):
    cids = []
    smiles = []
    names = []
    with tqdm(total=len(cas)) as pbar:  
        for c in cas:
            if pd.isnull(c) is not True:
                cid,smile,name = jscrawler(c)
            else:
                cid,smile,name = 'None','None','None'
            cids.append(cid)
            smiles.append(smile)
            names.append(name)
            pbar.update(1)
    return cids,smiles,names
    
def main():
    data,cas = dataloader()
    cas = cas[:5]
    cids,smiles,names = datacheker(cas)
    return data,cids,smiles,names

if __name__ == '__main__':
    data,cids,smiles,names = main()

    