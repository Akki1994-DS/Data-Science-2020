# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:14:27 2020

@author: axays
"""


import pandas as pd
import numpy as np

Company=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\Company_Data.csv')

Company.head()
Company.describe()
Company.columns
Company.isnull().sum()
Company.nunique()

Company.loc[Company.Sales >=7.49,'sales']='High'
Company.loc[Company.Sales <7.49,'sales']='Low'

Company.drop('Sales',1,inplace=True)

Company.head()
Company.ShelveLoc.value_counts()
Company.sales.value_counts()
Company.dtypes

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

Company.Urban=le.fit_transform(Company.Urban)
Company.ShelveLoc=le.fit_transform(Company.ShelveLoc)
Company.US=le.fit_transform(Company.US)
Company.sales=le.fit_transform(Company.sales)

Colnames=list(Company.columns)
Predictors=Colnames[0:10]
Target=Colnames[10]

from sklearn.model_selection import train_test_split
train,test=train_test_split(Company, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
Model=DecisionTreeClassifier(criterion='entropy')
Model.fit(train[Predictors],train[Target])

Predictions=Model.predict(test[Predictors])

pd.Series(Predictions).value_counts()
pd.crosstab(test[Target], Predictions)


