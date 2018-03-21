#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 17:16
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : CNN.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import torch as t
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import visdom
import os

uni = "P35968"
path = ".\\models\\"

vis = visdom.Visdom()
vis.text(u'''<h1>CNN</h1>''',win='log',opts={'title':'nn accuracy'})

######## 超参数 #########
MAXLEN = 120
BATCH_SIZE = 512
EPOCH = 200
#WEIGHT_DECAY = 1e-5
#N_HIDDEN = 128
DROPOUT = 0.5

def read(d):
    data = pd.read_table(path+uni+"_"+d+".txt",delimiter='\t')
    smiles = data['smi']
    data['label'] = 0
    for i in range(len(data.index)):
        if data.iloc[i,3] < 1000:
            data.iloc[i,len(data.columns)-1] = 1
        else:
            data.iloc[i,len(data.columns)-1] = 0
    y = data.iloc[:,len(data.columns)-1]
    X_train,X_test,y_train,y_test = train_test_split(smiles,y,test_size=0.2,random_state=7)
    return smiles,X_train,X_test,y_train,y_test

def resplit(data):
    smiles = []
    for smile in data:
       smile = '~'.join(smile)
       smile = smile.replace('C~l','Cl')
       smile = smile.replace('B~r','Br')
       smile = smile.replace('S~i','Other')
       smile = smile.replace('S~e','Other')
       smile = smile.split('~')
       smiles.append(smile)
    return smiles

def gather_stat(smiles):
    lengths = []
    for x in smiles:
        lengths.append(len(x))    
    maxlen = np.max(lengths)
    
    # 统计数据
    lengths = np.array(lengths)
    print('max:',lengths.max())
    print('min:',lengths.min())
    print('mean:',lengths.mean())
    print('mediam:',np.median(lengths))
    plt.hist(lengths, bins='auto')
    return maxlen

def del_oversize(X_train,X_test,y_train,y_test):
    for index,item in enumerate(X_train):
        if len(item) > MAXLEN:
            del X_train[index]
            y_train = y_train.drop(y_train.index[index])
    
    for index,item in enumerate(X_test):
        if len(item) > MAXLEN:
            del X_test[index]
            y_test = y_test.drop(y_test.index[index])
    return X_train,X_test,y_train,y_test
        
def onehot_encode(maxlen,data):
    outer = []
    for smile in data:
        inner = []
        for _word in smile:
            if _word in word2ix.keys():
                new_data = word2ix[_word]
            else:
                new_data = 22
            inner.append(new_data)
        outer.append(inner)
    
    for i in outer:
        i.extend((maxlen-len(i)) * (42,))
    
    outer_onehot = []
    for j in outer:
        onehot_encoded = []
        for k in j:
            letter = [0 for _ in range(len(word2ix))]
            letter[k] = 1
            onehot_encoded.append(letter)
        onehot_encoded = np.array(onehot_encoded)
        onehot_encoded = onehot_encoded.T
        onehot_encoded = np.delete(onehot_encoded,-1,0)
        outer_onehot.append(onehot_encoded)
    outer_onehot = np.array(outer_onehot)
    return outer_onehot

def onehot_decode(data):
    smiles_ix = []
    data = data.reshape(data.shape[0],len(word2ix)-1,MAXLEN,order='F')
    for n in data:
        smile_ix = []
        for m in n.T:
            if m.sum() > 0:
                index = int(np.where(m==1)[0])
            else:
                index = 42
            smile_ix.append(index)
        smiles_ix.append(smile_ix)
    smiles_word = []
    for l in smiles_ix:
        p = [ix2word[ii] for ii in l]
        q = "".join(p)
        smiles_word.append(q)
    return smiles_word

def create_loader():
    torch_dataset = Data.TensorDataset(data_tensor=X_train, target_tensor=y_train)
    loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
    return loader

if os.path.exists('smile.npz'):
    dic_data = np.load('smile.npz')
    word2ix,ix2word  = dic_data['word2ix'].item(),dic_data['ix2word'].item()

if os.path.exists('data_7.npz'):
    cnn_data = np.load('data_7.npz')
    X_train,X_test,y_train,y_test  = cnn_data['X_train'],cnn_data['X_test'],cnn_data['y_train'],cnn_data['y_test']
else:
    raw_smiles,X_train,X_test,y_train,y_test = read('maccs')
    raw_smiles,X_train,X_test = resplit(raw_smiles),resplit(X_train),resplit(X_test)
    
    X_train,X_test,y_train,y_test = del_oversize(X_train,X_test,y_train,y_test)
    
    #maxlen = gather_stat(raw_smiles)
    
    X_train,X_test = onehot_encode(MAXLEN,X_train),onehot_encode(MAXLEN,X_test)
    
    # 增加深度
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    y_train,y_test = y_train.values,y_test.values

#N_FEATURE = X_train.shape[1]

#rf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
#answer_rf = rf.predict(X_test)
#accuracy = metrics.accuracy_score(y_test,answer_rf)
#print('随机森林：',str(accuracy))

X_train,X_test,y_train,y_test = t.from_numpy(X_train).type(t.FloatTensor),t.from_numpy(X_test).type(t.FloatTensor),t.from_numpy(y_train),t.from_numpy(y_test)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 42, 120)
            nn.Conv2d(
                in_channels=1,   # input height
                out_channels=16,   # n_filters
                kernel_size=3,    # filter size
                stride=1,       # filter movement/step
                padding=1,  # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),        # output shape (16, 42, 120)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # 在 2x2 空间里向下采样, output shape (16, 17, 60)
        )
        self.conv2 = nn.Sequential(   # shape (16, 21, 60)
            nn.Conv2d(16,32,3,1,1),    # (32, 21, 60)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),     # (32, 10, 30)
        )
        self.conv3 = nn.Sequential(   # shape (32, 10, 30)
            nn.Conv2d(32,64,3,1,1),    # (64, 10, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),     # (64, 5, 15)
        )
        self.fc = nn.Sequential(nn.Linear(64 * 5 * 15, 512),
                                nn.Dropout(DROPOUT),
                                nn.ReLU(),
                                nn.Linear(512,64),
                                nn.Dropout(DROPOUT),
                                nn.ReLU(),
                                nn.Linear(64,2)
                                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)      # (batch, 32, 9, 30)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)     # (batch, 32 * 9 * 30)
        output = self.fc(x)
        return output

net=CNN()
print(net)

optimizer = t.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
test_x,test_y = Variable(X_test), Variable(y_test)
train_loader = create_loader()
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
