#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Apr 20 16:12:41 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import pandas as pd
import pybel
import openbabel as ob
from rdkit import Chem
from rdkit.Chem.inchi import rdinchi

def smitosmile(smi):
    mol = pybel.readstring("smi", smi)
    smile = mol.write('can')
    smile = smile.replace('\t\n', '')
    return smile
    
def obsmitosmile(smi):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile

def smiletoinchikey(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        smi = smitosmile(smile)
        mol = Chem.MolFromSmiles(smi)
    inchi = rdinchi.MolToInchi(mol)
    inchikey = rdinchi.InchiToInchiKey(inchi[0])
    return inchikey


data = pd.read_table('done_wash_wash_de2_can.csv',sep=',')

#转inchi
rd_inchi = []
for smile in data['can']:
    print(smile)
    inchikey = smiletoinchikey(smile)
    rd_inchi.append(inchikey)
data['rd_inchi'] = rd_inchi

#统一为can
#can = []
#for smile in data['Ligand SMILES']:
#    if smile == 'O=BOB(OB(OB=O)[O-])[O-]':
#        csmi = smile
#    elif smile == 'O=BO[B-](=O)O[B-](OB=O)=O':
#        csmi = smile
#    else:
#        csmi = smitosmile(smile)
#    can.append(csmi)
#data['can'] = can

#保存
data.to_csv('done_wash_wash_de2_inchi.csv',index=False)

##检查错误
#wrong = []
#for i in range(205709):
#    if data.iloc[i,5] != data.iloc[i,6]:
#        wrong.append(i)
        