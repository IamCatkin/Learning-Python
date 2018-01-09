# -*- coding: utf-8 -*-
import os
import pandas as pd

path = ".\\results\\"
files_0 = os.listdir(path)

########### 用来排除文件夹  ##############
files = []
for f in files_0:
    if os.path.isfile(path+f):
        files.append(f)
for file in files:
    filename = file[:-4]
    lc = pd.DataFrame(pd.read_csv(path+filename+".csv",header=0))
    sorted_lc = lc.sort_values(["ACC"],ascending=False).head(50)
    sorted_lc.to_csv(path+filename+'_sorted.csv',index=False)
