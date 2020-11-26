# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:41:33 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

company=pd.read_csv('C:/Users/HP/Desktop/assignments submission/decision tree/Company_Data.csv')
company.columns
company.Sales.median()
company.isna().sum()

#create bins for sales
cut_labels=['Low','Medium','High']
cut_bins=[-1,5.66,12,17]
company['sales']=pd.cut(company['Sales'],labels=cut_labels,bins=cut_bins)

company.pop('Sales')

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
company['ShelveLoc']=label_encoder.fit_transform(company['ShelveLoc'])
company['Urban']=label_encoder.fit_transform(company['Urban'])
company['US']=label_encoder.fit_transform(company['US'])

col_names=list(company.columns)
predictors=col_names[0:10]
target=col_names[10]

from sklearn.model_selection import train_test_split
train,test=train_test_split(company,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier as DS
model=DS(criterion='entropy')
model.fit(train[predictors],train[target])
train_pred=model.predict(train[predictors])
test_pred=model.predict(test[predictors])

train_acc=np.mean(train_pred==train[target])
test_acc=np.mean(test_pred==test[target])
train_acc#1.0
test_acc#0.66
