# Copy input to output
import pandas as pd
output_table = input_table.copy()
lists = set([i for i in output_table["Uniprot_ID"]])
index = []
for item in lists:
    x = output_table[output_table["Uniprot_ID"]==item]
    p = x.iloc[:,5].argmax()
    if p == p:
        index.append(p)
output_table = output_table.loc[index]