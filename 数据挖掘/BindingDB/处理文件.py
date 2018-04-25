import pandas as pd

def read():
    data = pd.read_table('BindingDB_All.tsv', sep='\t',encoding='utf-8', engine='c',error_bad_lines=False) 
    index = ["BindingDB Reactant_set_id","Ligand SMILES","Ki (nM)","BindingDB Target Chain  Sequence","UniProt (SwissProt) Primary ID of Target Chain"]
    data_a = data.loc[:,index]
    data_a.to_csv('trans_result.csv',sep='\t',index=False)

def dropna():
    data = pd.read_table('trans_result.csv', sep='\t') 
    data_b = data.dropna(axis=0)
    data_b.to_csv('all.csv',sep=',',index=False)

def filtrate():
    data = pd.read_table('all.csv', sep=',') 
    data_a = pd.read_table('20170609chembl_bindingdb_sum.csv', sep=',')
    unis = list(data_a['uni'].dropna(axis=0))
    data_b = data[data["UniProt (SwissProt) Primary ID of Target Chain"].isin(unis)]
    data_c = data_b[~data_b["Ki (nM)"].str.contains('<|>')]
    data_c.to_csv('done.csv',sep=',',index=False)
    
if __name__ == '__main__':
#    read()
#    dropna()
    filtrate()
    
    

