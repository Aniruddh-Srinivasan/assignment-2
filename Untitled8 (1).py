#!/usr/bin/env python
# coding: utf-8

# In[10]:


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:16:50 2021

@author: S.ANIRUDDH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import os
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn import metrics

print(os.getcwd())

## before working please ensure this file and data set are in same directory

tab1=pd.read_csv('train logis.csv',usecols=['Pclass','Age','Sex','SibSp','Parch','Embarked','Survived'])

label_X = LabelEncoder()
df2=tab1['Sex']
tab1['Sex']=label_X.fit_transform(df2)

label_X = LabelEncoder()
df2=tab1['Embarked']
tab1['Embarked']=label_X.fit_transform(df2)

mean1=tab1['Age'].mean()
tab1.loc[tab1['Age'].isna(),['Age']]=mean1

mean1=tab1['Embarked'].mode()
tab1.loc[tab1['Embarked'].isna(),['Embarked']]=mean1


X=tab1.iloc[:,[0,1,2,3,4,5]]
Y=tab1.iloc[:,[6]]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.9)
logis=LogisticRegression()
model=logis.fit(x_train,y_train)
y_predict=logis.predict(x_test)
cnf_matrix=metrics.confusion_matrix(y_test,y_predict)
print(cnf_matrix)


# In[ ]:




