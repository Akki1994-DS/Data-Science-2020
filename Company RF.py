# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:43:15 2020

@author: axays
"""

import pandas as pd
import numpy as np


Company=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\Company_Data RF.csv')

Company.describe()

Company.loc[Company.Sales >7.49,'sales']='High'
Company.loc[Company.Sales <=7.49,'sales']='Low'

Company.drop('Sales',1,inplace=True)

Company.head()
Company.ComPrice.value_counts()
Company.sales.value_counts()
Company.dtypes
Company.isnull().sum()

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

from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)
Forest=RandomForestClassifier(n_jobs=2,random_state=0)
Forest.fit(train[Predictors],train[Target])

Predictions=Forest.predict(test[Predictors])

pd.Series(Predictions).value_counts()
pd.crosstab(test[Target], Predictions)

#creating 500 Decision trees
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=500,criterion="entropy")
rf.fit(train[Predictors],train[Target])

Predictions1=Forest.predict(test[Predictors])

pd.Series(Predictions1).value_counts()
pd.crosstab(test[Target], Predictions1)






