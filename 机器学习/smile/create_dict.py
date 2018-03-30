#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 11:16
# @Author  : Catkin
# @Website : blog.catkin.moe
# @File    : create_dict.py
import pandas as pd
import numpy as np

uni = "P35968"
path = ".\\models\\"

def read(d):
    data = pd.read_table(path+uni+"_"+d+".txt",delimiter='\t')
    smiles = data['smi']
    return smiles

raw_smiles = read('cats')
smiles = []
for smile in raw_smiles:
   smile = '~'.join(smile)
   smile = smile.replace('C~l','Cl')
   smile = smile.replace('B~r','Br')
   smile = smile.split('~')
   smiles.append(smile)

words = {_word for _sentence in smiles for  _word in _sentence}
word2ix = {_word:_ix for _ix,_word in enumerate(words)}
word2ix[''] = len(word2ix) # 空格
ix2word = {_ix:_word for _word,_ix in list(word2ix.items())}

#np.savez_compressed('smile.npz',word2ix=word2ix,ix2word=ix2word)