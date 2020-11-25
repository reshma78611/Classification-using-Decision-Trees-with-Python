# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:50:02 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris_data=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/decision tree/iris.csv')

colnames=list(iris_data.columns)
predictors=colnames[0:4]
target=colnames[4]

from sklearn.model_selection import train_test_split
train,test=train_test_split(iris_data,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier as DS
model=DS(criterion='entropy')
model.fit(train[predictors],train[target])
train_pred=model.predict(train[predictors])
test_pred=model.predict(test[predictors])
#pd.crosstab(test_pred,test[target])

train_acc=np.mean(train_pred==train[target])
train_acc#100

test_acc=np.mean(test_pred==test[target])
test_acc#100
