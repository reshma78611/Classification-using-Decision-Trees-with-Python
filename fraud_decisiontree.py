# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 23:30:29 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fraud=pd.read_csv('C:/Users/HP/Desktop/assignments submission/decision tree/Fraud_check.csv')
fraud.columns
fraud.columns=['under_grad','marital_status','taxable_income','city_pop','work_exp','urban']
#creating bins for taxable_income=>to categorical
cut_labels=['Risky','Good']
cut_bins=[0,30000,99620]
fraud['tax_inc']=pd.cut(fraud['taxable_income'],bins=cut_bins,labels=cut_labels)
fraud.pop('taxable_income')

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
fraud['under_grad']=label_encoder.fit_transform(fraud['under_grad'])
fraud['marital_status']=label_encoder.fit_transform(fraud['marital_status'])
fraud['urban']=label_encoder.fit_transform(fraud['urban'])

col_names=list(fraud.columns)
predictors=col_names[0:5]
target=col_names[5]

from sklearn.model_selection import train_test_split
train,test=train_test_split(fraud,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier as DS
model=DS(criterion='entropy')
model.fit(train[predictors],train[target])
train_pred=model.predict(train[predictors])
test_pred=model.predict(test[predictors])

train_acc=np.mean(train_pred==train[target])
test_acc=np.mean(test_pred==test[target])
train_acc#1.0
test_acc#0.53
