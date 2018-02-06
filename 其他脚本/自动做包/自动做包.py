#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/6 17:50
# @Author  : Catkin
# @Site    : catkin.moe
# @File    : steam登陆.py
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time

boosterlist = ["Aselia the Eternal -The Spirit of Eternity Sword-","Astebreed: Definitive Edition","Beats Fever","Dream Of Mirror Online","Pure Hold'em",
               "Echo of Soul","Flowers -Le volume sur printemps-","Mu Complex","Niko: Through The Dream","Time Rifters","Franchise Hockey Manager 2014","SparkDimension"]
driver = webdriver.Chrome()

def getcookielist():
    driver.get("https://steamcommunity.com//tradingcards/boostercreator/")
    time.sleep(3)
    driver.find_element_by_id("steamAccountName").send_keys("账号")
    driver.find_element_by_id("steamPassword").send_keys("密码")
    time.sleep(3)
    driver.find_element_by_id("SteamLogin").click()
    key = input("请输入手机验证码: ")
    driver.find_element_by_id("twofactorcode_entry").send_keys(key)
    driver.find_element_by_css_selector("#login_twofactorauth_buttonset_entercode > div.auth_button.leftbtn > div.auth_button_h3").click()
    time.sleep(5)  
    driver.get("https://steamcommunity.com//tradingcards/boostercreator/")
    cookielist = driver.get_cookies()
    with open('steam.txt','w') as f:
        for item in cookielist:
            f.write(str(item)+'\n')
    return cookielist

def addcookies():    
    driver.get("https://steamcommunity.com//tradingcards/boostercreator/")
    with open('steam.txt','r') as f:
        result = []
        for line in f:
            item = eval(line)
            result.append(item)
    for cookie in result:
        driver.add_cookie(cookie)

def openbooster():
    driver.find_element_by_css_selector("div.btnv6_blue_blue_innerfade.btn_medium.btn_makepack > span").click()
    time.sleep(1)
    driver.find_element_by_css_selector("div.btn_green_white_innerfade.btn_medium > span").click()
    time.sleep(5)
    driver.find_element_by_css_selector("div.btn_grey_grey.btn_small > span").click()
    time.sleep(5)
    driver.find_element_by_css_selector("div.btn_grey_white_innerfade.btn_medium.booster_unpack_closebtn > span").click()
    time.sleep(1)

def select(name):
    Select(driver.find_element_by_id("booster_game_selector")).select_by_visible_text(name)

def createbooster():
    driver.get("https://steamcommunity.com//tradingcards/boostercreator/")
    for name in boosterlist:
        select(name)
        openbooster()
    driver.close()
    
# cookielist = getcookielist()