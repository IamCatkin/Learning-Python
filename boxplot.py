# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

#添加成绩表
plt.style.use("classic")

#新建一个空的DataFrame
df=pd.DataFrame()

#读取数据
seq = 1
while seq < 23:
    path =r"D:\python\results\\"
    data = pd.read_table(path+str(seq)+'.csv',delimiter=',')
    df[str(seq)]=data.iloc[:,[0]]
    seq += 1

#用matplotlib来画出箱型图
acc = plt.boxplot(x=df.values,
            patch_artist=True,
            boxprops = {'color':'black','facecolor':'#9999ff'},
            labels=df.columns,
            whis=1.5)
plt.title('The Accuracy boxplot')
plt.ylabel('Accuracy')
plt.xlabel('Hepatotoxicity class')
plt.show()