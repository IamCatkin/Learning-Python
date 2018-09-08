#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Sep  6 19:09:00 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import time,datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

#parameter
descriptors = ['cats','moe2d']
PRECISION = 2

# define a function to generate the current time string 
def timenow():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# load data
def dataloader(descriptor):
    if descriptor == 'cats':
        path = 'cats/'
        targets = pd.read_table(path+'targets_cansmiles_wash_cats.txt')
        asinex = pd.read_table(path+'asinex_cansmiles_wash_cats.txt')
        chembl = pd.read_table(path+'chembl_cansmiles_wash_cats.txt')
        np_cansmiles = pd.read_table(path+'np_cansmiles_wash_cats.txt')
        specs = pd.read_table(path+'specs_cansmiles_wash_cats.txt')
    if descriptor == 'moe2d':
        path = 'moe2d/'
        targets = pd.read_table(path+'targets_cansmiles_wash_moe2d.txt',sep=',')
        asinex = pd.read_table(path+'asinex_cansmiles_wash_moe2d.txt',sep=',')
        chembl = pd.read_table(path+'chembl_cansmiles_wash_moe2d.txt',sep=',')
        np_cansmiles = pd.read_table(path+'np_cansmiles_wash_moe2d.txt',sep=',')
        specs = pd.read_table(path+'specs_cansmiles_wash_moe2d.txt',sep=',')
    return targets,asinex,chembl,np_cansmiles,specs

# standardization  
def norm(data_a,data_b):
    data_a = data_a.iloc[:,3:]
    data_b = data_b.iloc[:,3:]
    length = len(data_a)
    data_all = pd.concat([data_a,data_b],axis=0)
    ss = StandardScaler()
    data_n = ss.fit_transform(data_all)
    data_na = data_n[:length,:]
    data_nb = data_n[length:,:]
    return data_na,data_nb

def main():
    start = time.time()
    for descriptor in descriptors:
        print("[{}]: Loading {} data ...".format(timenow(),descriptor))
        targets,asinex,chembl,np_cansmiles,specs = dataloader(descriptor)
      
        print("[{}]: calculating asinex data ...".format(timenow()))
        n_targets,asinex = norm(targets,asinex)
        s_matrix = cdist(asinex,n_targets)
        s_matrix = np.around(s_matrix,decimals=PRECISION)
        pd_matrix = pd.DataFrame(s_matrix)
        pd_matrix.to_csv('asinex_'+descriptor+'.txt',sep='\t',header=False,index=False)
        ss_matrix = cdist(n_targets,n_targets)
        ss_matrix = np.around(ss_matrix,decimals=PRECISION)
        spd_matrix = pd.DataFrame(ss_matrix)
        spd_matrix.to_csv('asinex_self_'+descriptor+'.txt',sep='\t',header=False,index=False)
        
        print("[{}]: calculating chembl data ...".format(timenow()))
        n_targets,chembl = norm(targets,chembl)
        s_matrix = cdist(chembl,n_targets)
        s_matrix = np.around(s_matrix,decimals=PRECISION)
        pd_matrix = pd.DataFrame(s_matrix)
        pd_matrix.to_csv('chembl_'+descriptor+'.txt',sep='\t',header=False,index=False)
        ss_matrix = cdist(n_targets,n_targets)
        ss_matrix = np.around(ss_matrix,decimals=PRECISION)
        spd_matrix = pd.DataFrame(ss_matrix)
        spd_matrix.to_csv('chembl_self_'+descriptor+'.txt',sep='\t',header=False,index=False)
        
        print("[{}]: calculating np_cansmiles data ...".format(timenow()))
        n_targets,np_cansmiles = norm(targets,np_cansmiles)
        s_matrix = cdist(np_cansmiles,n_targets)
        s_matrix = np.around(s_matrix,decimals=PRECISION)
        pd_matrix = pd.DataFrame(s_matrix)
        pd_matrix.to_csv('np_cansmiles_'+descriptor+'.txt',sep='\t',header=False,index=False)
        ss_matrix = cdist(n_targets,n_targets)
        ss_matrix = np.around(ss_matrix,decimals=PRECISION)
        spd_matrix = pd.DataFrame(ss_matrix)
        spd_matrix.to_csv('np_cansmiles_self_'+descriptor+'.txt',sep='\t',header=False,index=False)
        
        print("[{}]: calculating specs data ...".format(timenow()))
        n_targets,specs = norm(targets,specs)
        s_matrix = cdist(specs,n_targets)
        s_matrix = np.around(s_matrix,decimals=PRECISION)
        pd_matrix = pd.DataFrame(s_matrix)
        pd_matrix.to_csv('specs_'+descriptor+'.txt',sep='\t',header=False,index=False)
        ss_matrix = cdist(n_targets,n_targets)
        ss_matrix = np.around(ss_matrix,decimals=PRECISION)
        spd_matrix = pd.DataFrame(ss_matrix)
        spd_matrix.to_csv('np_cansmiles_self_'+descriptor+'.txt',sep='\t',header=False,index=False)
        
    end = time.time()
    print("[{}]: Mission completed,{} seconds used.".format(timenow(),round(end-start))) 

if __name__ == '__main__':
    main()

