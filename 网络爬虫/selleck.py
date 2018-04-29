#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Sun Apr 29 15:12:32 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as pd
import requests

names = []
contents = []
data = pd.read_excel('123.xlsx')
for item in data['name']:
    url = 'http://www.selleck.cn/search.html?searchDTO.searchParam='+item+'&sp='+item
    page = requests.get(url).text
    table = pd.read_html(page)[0]
    name = table.iloc[1,2]
    content = table.iloc[1,3]
    names.append(name)
    contents.append(content)
data['names'] = names
data['contents'] = contents
data.to_csv('123.csv',index=False)