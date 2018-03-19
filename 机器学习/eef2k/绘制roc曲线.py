import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt  
data = pd.read_table('gold_PLP.csv',delimiter=',')
#score_mean = data['Goldscore.Fitness'].mean()
#score_max = data['Goldscore.Fitness'].max()
#score_min = data['Goldscore.Fitness'].min()
#score_dif = score_max-score_min
#data['pre_score'] = None
#for i in range(len(data.index)):
#    data.iloc[i,2]=(data.iloc[i,1]-score_min)/score_dif
y_test = data.iloc[:,0]
y_score = data.iloc[:,1]
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_score,pos_label=1)
auc = metrics.auc(fpr,tpr)
print(auc)

roc = plt.figure() 
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.plot(fpr,tpr,linewidth=2,label='ROC')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,1.0)
plt.ylim(0,1.05)
plt.legend(loc=4)#图例的位置
plt.show()
roc.savefig('roc.pdf')