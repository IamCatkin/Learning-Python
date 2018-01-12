# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt  

#########  建立空dataframe及表   ##########
plt.style.use("classic")
df = pd.DataFrame()

#######     读取数据   #########
sorted_path = ".\\results\\cats\\"
sortedfiles_0 = os.listdir(sorted_path)
sortedfiles = []
for f0 in sortedfiles_0:
    if os.path.isfile(sorted_path+f0):
        if ".csv" in f0:
            sortedfiles.append(f0)
for sortedfile in sortedfiles:
    sortedfilename = sortedfile[0:6]
    data = pd.read_table(sorted_path+sortedfilename+'_sorted.csv',delimiter=',')
    df[sortedfilename]=data.iloc[:,[3]]

##########   用matplotlib来画出箱型图   #######
acc = plt.figure() 
plt.boxplot(x=df.values,
            patch_artist=True,
            boxprops={'color':'black','facecolor':'#9999ff'},
            labels=df.columns,
            whis=1.5)
plt.title('The auc boxplot')
plt.ylabel('auc')
plt.xlabel('Hepatotoxicity class')
plt.show()

#######    保存图片  ############
acc.savefig(sorted_path+"auc.pdf", bbox_inches='tight')
                       
