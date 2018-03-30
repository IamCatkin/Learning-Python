#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 15:44
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : MLP.py
import os
import visdom
import math
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics
#########     参数     ###########
path = '.\\data\\'
cycles = 20

MAXLEN = 120
VAR = 1e-6
LR = 1e-3
BATCH_SIZE = 256
EPOCH = 300
WEIGHT_DECAY = 1e-5
N_HIDDEN = 128
DROPOUT = 0.5

class MLP(nn.Module):
    def __init__(self,n_feature, n_hidden, dropout):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(n_feature, n_hidden),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.Linear(n_hidden,n_hidden),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.Linear(n_hidden,2)
                                )
    def forward(self, x):
        output = self.fc(x)
        return output
    
def getproteins():
    files = os.listdir(path)  
    names = set()
    for file in files:
        if '.csv' in file:
            name = file[0:file.rfind('-', 1)]
            names.add(name)
    names = list(names)
    return names

def check_up(data):
    dellist = []
    for item in range(data.shape[1]):
        data0 = data[:,item]
        data0_var = data0.var()
        if data0_var < VAR:
            dellist.append(item)
    return dellist

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
    lengths = np.array(lengths)
    print('max:',lengths.max())
    print('min:',lengths.min())
    print('mean:',lengths.mean())
    print('mediam:',np.median(lengths))
    plt.hist(lengths, bins='auto')
    return maxlen

def del_oversize(data_x,data_y,maxlen):
    dellist=[]
    for index,item in enumerate(data_x):
        if len(item) > maxlen:
            dellist.append(index)
    data_x = [x for x in data_x if len(x) <= maxlen]
    dellist = [data_y.index[i] for i in dellist]
    data_y = data_y.drop(dellist)    
    return data_x,data_y
        
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

def create_loader(X_train,y_train):
    torch_dataset = Data.TensorDataset(data_tensor=X_train, target_tensor=y_train)
    loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
    return loader

def model(protein,i):
    if i == 0:
        with open('.\\result\\'+protein+'-MLP.csv','a+') as f:
            f.write('ACC'+','+'SPE'+','+'SEN'+','+'AUC'+'\n')
    
    data = pd.read_table(path+protein+'-maccs.csv',delimiter=',')   
    X = data.iloc[:,0]
    y = data.iloc[:,2]
    X_,X_test,y_,y_test = train_test_split(X,y,test_size=0.2)
    X_train,X_validation,y_train,y_validation = train_test_split(X_,y_,test_size=0.25)
    X_train,X_validation,X_test = resplit(X_train),resplit(X_validation),resplit(X_test)
    X_train,y_train = del_oversize(X_train,y_train,MAXLEN)
    X_test,y_test = del_oversize(X_test,y_test,MAXLEN)
    X_validation,y_validation = del_oversize(X_validation,y_validation,MAXLEN)
    X_train,X_test,X_validation = onehot_encode(MAXLEN,X_train),onehot_encode(MAXLEN,X_test),onehot_encode(MAXLEN,X_validation)
    X_train,X_test,X_validation = X_train.reshape(X_train.shape[0],-1,order='F'),X_test.reshape(X_test.shape[0],-1,order='F'),X_validation.reshape(X_validation.shape[0],-1,order='F')
    y_train,y_test,y_validation = y_train.values,y_test.values,y_validation.values
    dellist = check_up(X_train)
    X_train = np.delete(X_train,dellist,1)
    X_test = np.delete(X_test,dellist,1)
    X_validation = np.delete(X_validation,dellist,1)
    N_FEATURE = X_train.shape[1]
    X_train,X_test,X_validation,y_train,y_test,y_validation = t.from_numpy(X_train).type(t.FloatTensor),t.from_numpy(X_test).type(t.FloatTensor),t.from_numpy(X_validation).type(t.FloatTensor),t.from_numpy(y_train),t.from_numpy(y_test),t.from_numpy(y_validation)
    net=MLP(N_FEATURE,N_HIDDEN,DROPOUT)
    if i == 0:
        print(net)
    optimizer = t.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_func = nn.CrossEntropyLoss()
    test_x,test_y,validation_x,validation_y = Variable(X_test), Variable(y_test), Variable(X_validation), Variable(y_validation)
    train_loader = create_loader(X_train,y_train)
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
            
            if step == math.floor(X_train.size(0)/BATCH_SIZE):
                net.eval()  #有dropout时，预测时转换模式，把dropout断掉
                validation_output = net(validation_x)
                score = F.softmax(validation_output,dim=1)[:,1].data.numpy()
                fpr,tpr,thresholds = metrics.roc_curve(y_validation.numpy(),score,pos_label=1)
                auc = metrics.auc(fpr,tpr)
                pred_y = t.max(F.softmax(validation_output,dim=1), 1)[1]
                accuracy = (pred_y == validation_y).data.numpy().sum() / validation_y.size(0)
                text = 'Epoch:&nbsp;' + str(epoch) + '&nbsp;&nbsp;|&nbsp;acc: %.4f' % accuracy + '&nbsp;&nbsp;|&nbsp;auc: %.4f' % auc
                x_1 = t.Tensor([epoch])
                y_3 = t.Tensor([loss.data[0]])  #交叉熵损失
                y_1 = t.Tensor([accuracy_t_y])
                y_2 = t.Tensor([accuracy])
                y_4 = t.Tensor([auc])
                vis.line(X=x_1,Y=y_3,win='pic1',update='append' if epoch >0 else None,opts=dict(title='acc & loss'))
                vis.updateTrace(X=x_1, Y=y_1,win='pic1',name='train')
                vis.updateTrace(X=x_1, Y=y_2,win='pic1',name='validation')
                vis.updateTrace(X=x_1, Y=y_4,win='pic1',name='auc')
                vis.text(text,win='log',opts={'title':'nn accuracy'},append=True)
                net.train()
    
    net.eval()
    test_output = net(test_x)
    score = F.softmax(test_output,dim=1)[:,1].data.numpy()
    fpr,tpr,thresholds = metrics.roc_curve(y_test.numpy(),score,pos_label=1)
    auc = metrics.auc(fpr,tpr)
    pred_y = t.max(F.softmax(test_output,dim=1), 1)[1]
    acc = (pred_y == test_y).data.numpy().sum() / test_y.size(0)
    confusion = metrics.confusion_matrix(y_test.numpy(),pred_y.data.numpy())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    sen = TP / (TP+FN)
    spe = TN / (TN+FP)
    print('=========='+protein+'-MLP=========')
    print('The accuracy is: %.3f' %acc)
    print('The specificity is: %.3f' %spe)
    print('The sensitivity is: %.3f' %sen)
    print('The auc is: %.3f' %auc)
    print('============================')        

    with open('.\\result\\'+protein+'-MLP.csv','a+') as f:
        f.write('{:.3f}'.format(float(acc))+','+'{:.3f}'.format(float(spe))+','+'{:.3f}'.format(float(sen))+','+'{:.3f}'.format(float(auc))+'\n')

if __name__ == '__main__':
    if os.path.exists('smile.npz'):
        dic_data = np.load('smile.npz')
        word2ix,ix2word  = dic_data['word2ix'].item(),dic_data['ix2word'].item()
    proteins = getproteins()
    for protein in proteins:
        for i in range(cycles):
            vis = visdom.Visdom()
            vis.text(u'''<h1>MLP</h1>''',win='log',opts={'title':'nn accuracy'})
            print('=========== '+ str(i) +' ===========')
            model(protein,i)
