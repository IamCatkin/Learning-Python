#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 10:00
# @Author  : Catkin
# @Website : blog.catkin.moe

import pybel
import pandas as  pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Scaffolds import MurckoScaffold

def smiToSmile(smi):
    mol = pybel.readstring("smi", smi)
    smile = mol.write('can')
    smile = smile.replace('\t\n', '')
    return smile

def addCan(data):
    can = []
    for smile in data['smi']:
        if smile == 'O=BOB(OB(OB=O)[O-])[O-]':
            csmi = smile
        elif smile == 'O=BO[B-](=O)O[B-](OB=O)=O':
            csmi = smile
        else:
            csmi = smiToSmile(smile)
        can.append(csmi)
    data['can'] = can

    data.to_csv('smi_n.csv',index=False)

def computeFramwork(df):
    murckos = []
    carbons = []
    for smi in df['can']:
        mol = Chem.MolFromSmiles(smi)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        carb = MurckoScaffold.MakeScaffoldGeneric(core)
        #将Murcko骨架和C骨架转成smile
        mur = Chem.MolToSmiles(core)
        carb = Chem.MolToSmiles(carb)
        murckos.append(mur)
        carbons.append(carb)
    df['murckos'] = murckos
    df['carbons'] = carbons
    return df

def getsmi():
    data = pd.read_table('bingdingdb_2_2_4.csv',sep=',')
    s = set()
    for i in data['smi']:
        s.add(i)
    df = pd.DataFrame()
    df['smi'] = list(s)
    df.to_csv('smi.csv',index=True)
    
def checknan():
    df = pd.read_table('smi_moe.csv',sep=',')
    na = df[df.isnull().values==True]
    na = na.drop_duplicates()
    return na

def getuniport():
    data = pd.read_table('bingdingdb_2_2_4.csv',sep=',')
    seq = data['BindingDB Target Chain  Sequence']
    uni = data['UniProt (SwissProt) Primary ID of Target Chain']
    dic = dict(zip(uni, seq))
    result = pd.DataFrame()
    result['uni'] = dic.keys()
    result['seq'] = dic.values()
    result.to_csv('uni.csv', index=True)

def toindex():
    data = pd.read_table('bingdingdb_2_2_4.csv',sep=',')
    result = data[['smi','UniProt (SwissProt) Primary ID of Target Chain','Ki (nM)']]
    smi = pd.read_table('smi.csv',sep=',')
    uni = pd.read_table('uni.csv',sep=',')
    s_dic = dict(zip(smi['smi'],smi['index']))
    u_dic = dict(zip(uni['uni'],uni['index']))
    result1 = result.copy()
    for i in range(len(result)):
        s = s_dic[result.iloc[i,0]]
        result1.iloc[i,0] = s
        u = u_dic[result.iloc[i,1]]
        result1.iloc[i,1] = u
    result1.to_csv('s-u.csv', index=False)
    
if __name__ == '__main__':
    pass