#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cairosvg
filename = input('Please enter filename:>>')
cairosvg.svg2pdf(
    file_obj=open(filename+'.svg', "rb"), write_to=filename+'.pdf')
input('请按任意键退出')