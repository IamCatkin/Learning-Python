#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2017/11/2 14:13
# @Author  : Catkin
# @File    : buhuo.py
import requests
import re
from time import sleep
from lxml import html
from email.mime.text import MIMEText
it = True
n = 0
while it == True:
    page = requests.get(url='https://www.pujia8.com/gifts/')
    tree = html.fromstring(page.text)
    item = tree.xpath('//*[@id="gift227"]/tbody/tr/td[3]/p[1]/text()[4]')
    item_1 = ''.join(item)
    patterns = re.compile('\u5269\u4f59(.*?)\u4e2a')
    some = patterns.findall(item_1)
    k = ['0']
    if some != k:
        msg = MIMEText('补货了速度去抢.', 'plain', 'utf-8')
        from_addr = 发件邮箱
        password = 邮箱密码
        smtp_server = smtp服务器
        to_addr = 收件邮箱
        import smtplib
        server = smtplib.SMTP(smtp_server, 25)
        server.set_debuglevel(0)
        server.login(from_addr, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()
        it = False
    from random import randint
    shijian = randint(60, 200)
    n += 1
    print('第'+str(n)+'次检测')
    sleep(shijian)