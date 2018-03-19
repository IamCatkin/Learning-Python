#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/7 13:46
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : pytroch-classification-2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import torch as t
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import visdom

vis = visdom.Visdom()
vis.text(u'''<h1>神经网络</h1>''',win='log',opts={'title':'nn accuracy'})
uni = "P35968"
path = ".\\models\\"

######## 超参数 #########
BATCH_SIZE = 128
EPOCH = 1000
VAR = 0.000001
WEIGHT_DECAY = 1e-5
N_HIDDEN = 128
DROPOUT = 0.5
#SGD
SGD_LR = 0.01
SGD_MOMENTUM = 0.9

#######################

def read(d):
    data = pd.read_table(path+uni+"_"+d+".txt",delimiter='\t')
    data['label'] = 0
    for i in range(len(data.index)):
        if data.iloc[i,3]<1000:
            data.iloc[i,len(data.columns)-1]=1
        else:
            data.iloc[i,len(data.columns)-1]=0
    X_0 = data.iloc[:,7:len(data.columns)-1]
    y_0 = data.iloc[:,len(data.columns)-1]    
    X_train,X_test,y_train,y_test = train_test_split(X_0,y_0,test_size=0.2,random_state=7)
    return X_train,X_test,y_train,y_test

def check_up(data):
    dellist = []
    for item in range(data.shape[1]):
        data0 = data[:,item]
        data0_var = data0.var()
        if data0_var < VAR:
            dellist.append(item)
    return dellist

def zero_normalize(data_train,data_test):
    for item in range(data_train.shape[1]):
        data_train_0 = data_train[:,item]
        data_test_0 = data_test[:,item]
        
        data_train_0_avg = data_train_0.mean()
        data_train_0_std = data_train_0.std(ddof=1)
        
        data_train_1 = (data_train_0-data_train_0_avg)/data_train_0_std
        data_test_1 = (data_test_0-data_train_0_avg)/data_train_0_std
        data_train[:,item]=data_train_1
        data_test[:,item]=data_test_1
    return data_train,data_test

def mm_normalize(data_train,data_test):
    for item in range(data_train.shape[1]):
        data_train_0 = data_train[:,item]
        data_test_0 = data_test[:,item]
        
        data_train_0_max = data_train_0.max()
        data_train_0_min = data_train_0.min()
        data_train_0_dif = data_train_0_max-data_train_0_min
        
        data_train_1 = (data_train_0-data_train_0_min)/data_train_0_dif
        data_test_1 = (data_test_0-data_train_0_min)/data_train_0_dif
        data_train[:,item]=data_train_1
        data_test[:,item]=data_test_1
    return data_train,data_test

def create_loader():
    torch_dataset = Data.TensorDataset(data_tensor=X_train_tensor, target_tensor=y_train_tensor)
    loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
    return loader

if __name__ == '__main__':
    X_train,X_test,y_train,y_test = read("cats")
    X_train_array,y_train_array,X_test_array,y_test_array = X_train.values,y_train.values,X_test.values,y_test.values
    dellist = check_up(X_train_array)
    X_train_array = np.delete(X_train_array,dellist,1)
    X_test_array = np.delete(X_test_array,dellist,1)
#    X_train_array,X_test_array = mm_normalize(X_train_array,X_test_array)
    X_train_array,X_test_array = zero_normalize(X_train_array,X_test_array)
    X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor = t.from_numpy(X_train_array).type(t.FloatTensor),t.from_numpy(y_train_array),t.from_numpy(X_test_array).type(t.FloatTensor),t.from_numpy(y_test_array)

    N_FEATURE = X_train_tensor.shape[1]
    net = t.nn.Sequential(
        t.nn.Linear(N_FEATURE, N_HIDDEN),
        t.nn.Dropout(DROPOUT),  
        t.nn.ReLU(),
        t.nn.Linear(N_HIDDEN, N_HIDDEN),
        t.nn.Dropout(DROPOUT), 
        t.nn.ReLU(),
        t.nn.Linear(N_HIDDEN, N_HIDDEN),
        t.nn.Dropout(DROPOUT), 
        t.nn.ReLU(),
        t.nn.Linear(N_HIDDEN, 2),
    )        
    print(net)
    optimizer = t.optim.SGD(net.parameters(), lr=SGD_LR, momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY)
#    optimizer = t.optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    test_x,test_y = Variable(X_test_tensor), Variable(y_test_tensor)
    train_loader = create_loader()
    
    rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
    answer_rf = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,answer_rf)
    print('随机森林：',str(accuracy))
    
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)  # batch x
            b_y = Variable(y)  # batch y
    
            output = net(b_x)  # cnn output
            pred_t_y = t.max(F.softmax(output,dim=1), 1)[1]
            accuracy_t_y = (pred_t_y == b_y).data.numpy().sum() / b_y.size(0)
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            
            if step % 200 == 0:
                net.eval()  #有dropout时，预测时转换模式，把dropout断掉
                test_output = net(test_x)
                pred_y = t.max(F.softmax(test_output,dim=1), 1)[1]
                accuracy = (pred_y == test_y).data.numpy().sum() / test_y.size(0)
                text = 'Epoch:&nbsp;' + str(epoch) + '&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;test accuracy: %.4f' % accuracy
                x_1 = t.Tensor([epoch])
                y_3 = t.Tensor([loss.data[0]])  #交叉熵损失
                y_1 = t.Tensor([accuracy_t_y])
                y_2 = t.Tensor([accuracy])
                vis.line(X=x_1,Y=y_3,win='pic1',update='append' if epoch >0 else None,opts=dict(title='acc & loss'))
                vis.updateTrace(X=x_1, Y=y_1,win='pic1',name='train')
                vis.updateTrace(X=x_1, Y=y_2,win='pic1',name='test')
                vis.text(text,win='log',opts={'title':'nn accuracy'},append=True)
                net.train()

# 读weight bias                
#params=net.state_dict() 
#for k,v in params.items():
#    print(k)    
#print(params['0.weight'])   
#print(params['0.bias'])  

# 保存加载模型
#t.save(net.state_dict(), 'net_1.pkl')
#net.load_state_dict(t.load('net_1.pkl'))