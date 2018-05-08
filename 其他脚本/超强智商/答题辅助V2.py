#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Tue May  8 11:36:10 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import json
import webbrowser

stat = True
while stat == True:
    try:
        file = open("d:\\python\\temp\\cqzs.txt","r",encoding="utf-8")
        j = json.loads(file.read())
        if 'game' in j['data']:
            question_list = j['data']['game']['question_list']
            for q in question_list:
                title = q['title']
                url = 'http://www.baidu.com/s?wd=%s' % title
                webbrowser.open(url)
            stat = False
        file.close()
    except:
        pass