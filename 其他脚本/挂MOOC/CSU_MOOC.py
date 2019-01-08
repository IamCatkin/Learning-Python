#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/01/08 13:29
# @Author  : Catkin
# @Website : blog.catkin.moe
import time,re
from retry import retry
from bs4 import BeautifulSoup
from selenium import webdriver

driver = webdriver.Firefox()
userid = ''
password = ''

@retry()
def login():
    driver.get('http://gra.csu.xuetangx.com/')
    driver.find_element_by_css_selector('div.col-md-2:nth-child(3) > ul:nth-child(1) > li:nth-child(1) > a:nth-child(1)').click()
    driver.find_element_by_css_selector('div.form-group:nth-child(3) > input:nth-child(1)').click()
    time.sleep(2)
    driver.find_element_by_css_selector('div.form-group:nth-child(3) > input:nth-child(1)').send_keys(userid)    
    driver.find_element_by_css_selector('div.form-group:nth-child(5) > input:nth-child(1)').click()
    time.sleep(2)
    driver.find_element_by_css_selector('div.form-group:nth-child(5) > input:nth-child(1)').send_keys(password)
    driver.find_element_by_css_selector('.btn').click()

@retry()
def get_courses():
    driver.find_element_by_css_selector('a.btn'
                                        ).click()   
    time.sleep(5)
    handles = driver.window_handles
    driver.switch_to_window(handles[-1])
    time.sleep(5)
    html = driver.page_source
    bsObj = BeautifulSoup(html, 'lxml')
    course_list = []
    for a in bsObj.find_all('li',class_=" graded")[:-1]:
        b = a.find('a')
        course_list.append(b.get('href'))
    return course_list

def get_time():
    time_html = driver.page_source
    cost = re.findall('<div class="xt_video_player_current_time_display fl"><span>(.*?)</span> / <span>(.*?)</span></div>', time_html, re.S)
    start = 60*int(cost[0][0].split(":")[0])+int(cost[0][0].split(":")[1])
    end = 60*int(cost[0][1].split(":")[0])+int(cost[0][1].split(":")[1])
    need = end-start
    return need

@retry()
def play(course,i):
    do = True
    while do is True:
        root = 'http://gra.csu.xuetangx.com'
        driver.get(root+course)
        time.sleep(10)
        need = get_time()
        if i == 0:
            driver.find_element_by_css_selector('.xt_video_player_volume_icon').click()
            js = 'document.getElementsByClassName("xt_video_player_common_list")[0].style.display="block";'
            driver.execute_script(js)
            driver.find_element_by_xpath('/html/body/div[3]/div[1]/div[2]/div/section/div/div/div[2]/div/div/div[1]/div/div/div/div[1]/div[1]/div[5]/ul/li[1]').click()
        driver.find_element_by_css_selector('.xt_video_player_play_btn').click()
        time.sleep((need/2)+3)
        time2 = get_time()
        if time2 == 0:
            return

def main():
    login()
    course_list = get_courses()
    i = 0
    for course in course_list[11:]:
        play(course,i)
        i += 1
        print('class:'+str(i)+',done!')

if __name__ ==  '__main__':
    main()