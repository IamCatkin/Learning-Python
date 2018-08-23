#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : Thu Aug  9 16:04:04 2018
# @Author  : Catkin
# @Website : blog.catkin.moe
import math
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split,cross_validate 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

#parameters
description = 'moe2d'

def dataloader():
    data = pd.read_csv('bcl2_'+description+'.csv')
    X = data.iloc[:,3:]
    y = data.iloc[:,2]
    y = -y.apply(math.log10)    
    return X,y

def rfmodel(X,y):
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
    rf = RandomForestRegressor(n_estimators = 500)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    scores = cross_validate(rf, X, y, cv=5, scoring=('r2', 'neg_mean_squared_error'))
    cv_mse = -scores['test_neg_mean_squared_error']
    cv_rmse = np.sqrt(cv_mse).mean()
    cv_r2 = scores['test_r2'].mean()
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
    print('RMSE train: %.3f, test: %.3f' % (math.sqrt(mean_squared_error(y_train, y_train_pred)),math.sqrt(mean_squared_error(y_test, y_test_pred))))
    print('CV R^2: %.3f, RMSE: %.3f' % (cv_r2,cv_rmse))
    return y_train,y_train_pred,y_test,y_test_pred

def datasaver():
    pass

def f_1(x, A, B):
    return A*x + B

def scatter(y_train,y_train_pred,y_test,y_test_pred):
    fig = plt.figure(figsize=(16, 12))
    x_1 = plt.scatter(y_train,y_train_pred,marker='^',color='b')
    x_2 = plt.scatter(y_test,y_test_pred,color='r')
    A1, B1 = optimize.curve_fit(f_1, y_test, y_test_pred)[0]
    x1 = y_test
    y1 = A1*x1 + B1
    line = plt.plot(x1, y1, 'k--', lw=3)
    plt.tick_params(labelsize=16)  
    plt.xlabel('Experimental Values',fontsize=20)
    plt.ylabel('Predicted Values',fontsize=20)
    plt.show()
    fig.savefig(description+'.pdf')
    plt.close()

def main():
    X,y = dataloader()
    y_train,y_train_pred,y_test,y_test_pred = rfmodel(X,y)
    scatter(y_train,y_train_pred,y_test,y_test_pred)
    
if __name__ == '__main__':
    main()