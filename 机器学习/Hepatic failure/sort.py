# -*- coding: utf-8 -*-
import pandas as pd

seq = 1
path = r"D:\python\results\\"
while seq < 23:
    lc = pd.DataFrame(pd.read_csv(path+str(seq)+'.csv',header=0))
    sorted_lc = lc.sort_values(["ACC"],ascending=False).head(20)
    sorted_lc.to_csv(path+str(seq)+'_sorted.csv',index=False)
    seq += 1