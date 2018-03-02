#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/1 22:54
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : fraunhofer-repdose.py
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

def login():
    header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/'
                      '537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    data = {'email':'your email',
            'password':'your password',
            'formsubmit':'Log In'}
    url = "http://fraunhofer-repdose.de/"
    lg = requests.post(url,data=data,headers=header)
    if "query_parameters.php" in lg.text:
        print("Login successful!")
    else:
        print("Login error")
        
def crawler(start, difference, maximum):
    try:
        result = pd.DataFrame()
        origin = "http://fraunhofer-repdose.de/repdose/"
        parameter1 = start
        parameter2 = parameter1 + difference
        while parameter1 < maximum:
            target = 'http://fraunhofer-repdose.de/repdose/query.php?cas_where=&cas_string=&cas_show=on&species=' \
                     '&species_show=on&organ=&organ_show=on&name=&name_show=on&s_sex=&ssex_show=on&effect=&effect_show=' \
                     'on&route=&route_show=on&e_sex=&esex_show=on&boilingpoint_c=&boilingpoint_show=on&duration_from=' \
                     '&duration_to=&duration_show=on&eloel_mg_from=&eloel_mg_to=&eloel_mg_show=on&watersolubility_c=&watersolubility_show' \
                     '=on&noel_mg_from=&noel_mg_to=&noel_mg_show=on&logpow_c=&logpow_show=on&loel_mg_from=&loel_mg_to=&loel_mg_show' \
                     '=on&pressure_c=&pressure_show=on&reliabilityA=on&reliabilityB=on&mol_from='+str(parameter1)+'&mol_to='+str(parameter2)+'&molweight_show=on&reference_show=0'
            page = requests.get(target).text
            if "Please restrict query conditions." in page:
                print(str(parameter1)+":error")
            elif "Page" in page:
                lists = []
                bsObj = BeautifulSoup(page, 'lxml')
                found_a = bsObj.find_all('a')
                for item in found_a:
                    found_href = item.get('href')
                    if "query.php" in found_href:
                        lists.append(found_href)
                for i in lists:
                    html = origin + i
                    r_page = requests.get(html).text
                    table = pd.read_html(r_page)[0]
                    table.drop([0,1], inplace=True)
                    result = pd.concat([result,table])
            else:
                table = pd.read_html(page)[0]
                table.drop([0,1], inplace=True)
                result = pd.concat([result,table])
            parameter1 += difference
            parameter2 += difference
            time.sleep(0.2)
    finally:
        result.to_csv("result_"+str(maximum)+".csv",index=False,header=False)

if __name__ == '__main__':
    start = input("Enter the min Mol. weight: ")
    difference = input("Enter the difference: ")
    maximum = input("Enter the max Mol. weight: ")
    login()
    crawler(int(start), int(difference), int(maximum))