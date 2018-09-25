#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Tue Sep 25 10:28:55 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import time
from tqdm import tqdm
from retry import retry
import pandas as pd
from lxml import html
from selenium import webdriver

def dataloader():
    data = pd.read_excel('database.xlsx')
    cid = data['CID']
    return data,cid

@retry()
def jscrawler(c):
    url = 'https://pubchem.ncbi.nlm.nih.gov/compound/' + c    
    fireFoxOptions = webdriver.FirefoxOptions()
    fireFoxOptions.set_headless()
    browser = webdriver.Firefox(firefox_options=fireFoxOptions)
    browser.get(url)
    time.sleep(5)
    page = browser.page_source
    browser.close()
    tree = html.fromstring(page)
    smile_0 = tree.xpath('//*[@id="Canonical-SMILES"]/div[2]/div[1]')
    if smile_0:
        smile = smile_0[0].text
        if smile is None:
            smile_1 = tree.xpath('//*[@id="Canonical-SMILES"]/div[2]/div[1]/span')
            if smile_1:
                smile = smile_1[0].text
    else:
        smile = 'None'
    return smile
        
def datacheker(cid):
    with open('result.csv','a+') as f:
        f.write("smile"+'\n')
        f.close()
    with tqdm(total=len(cid)) as pbar:  
        for c in cid:
            if pd.isnull(c) is not True:
                smile = jscrawler(c)
            else:
                smile = 'Nocid'
            with open('result.csv','a+') as f:
                f.write(smile+'\n')
                f.close()
            pbar.update(1)
    
def main():
    data,cid = dataloader()
    datacheker(cid)

if __name__ == '__main__':
    main()
