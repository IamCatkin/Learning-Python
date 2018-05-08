#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Wed May  2 13:19:42 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import win32gui, win32ui, win32con, win32api
from PIL import Image
import pytesseract
import webbrowser

def window_capture(filename):
    hwnd = 0
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    w = 480
    h = 120
    saveBitMap.CreateCompatibleBitmap(mfcDC,w,h)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0,0),(w,h),mfcDC,(50,320),win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC,filename)
window_capture('cqzs.jpg')
text=pytesseract.image_to_string(Image.open('cqzs.jpg'),lang='chi_sim')
new_text =''.join(text.split())
url = 'http://www.baidu.com/s?wd=%s' % new_text
webbrowser.open(url)