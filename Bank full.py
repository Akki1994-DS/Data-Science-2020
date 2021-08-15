# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:51:24 2020

@author: axays
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Bank_full=pd.read_csv("C:\\Users\\axays\Desktop\\bank-full (1).csv",sep=";")

#EDA
Bank_full.head()
Bank_full.columns
Bank_full.nunique()
Bank_full.describe()
Bank_full.isnull().sum()

Bank_full['default'].unique()
Bank_full['housing'].unique()
Bank_full['loan'].unique()
Bank_full['y'].unique()
Bank_full['marital'].unique()


#Created dummy variables for categorical values into binary format.
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

Bank_full.default=le.fit_transform(Bank_full.default)
Bank_full.housing=le.fit_transform(Bank_full.housing)
Bank_full.loan=le.fit_transform(Bank_full.loan)
Bank_full.y=le.fit_transform(Bank_full.y)

#Dropping unnecessary columns.
Bank_full=Bank_full.drop(['job'],axis=1)
Bank_full=Bank_full.drop(['marital'],axis=1)
Bank_full=Bank_full.drop(['education'],axis=1)
Bank_full=Bank_full.drop(['contact'],axis=1)
Bank_full=Bank_full.drop(['month'],axis=1)
Bank_full=Bank_full.drop(['poutcome'],axis=1)


#Making data normally distributed.
#Data is normally distributed because min value is 0 and max is 1.
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

Bank_norm=norm_func(Bank_full.iloc[:,0:])

Bank_norm.describe()


sns.countplot(x='age',data=Bank_norm)


sns.countplot(x='housing',hue='loan',data=Bank_norm)

#Defined input and output variables.
X=Bank_norm.iloc[:,[0,1,2,3,4,5,6,7,8,9]]
Y=Bank_norm.iloc[:,10]

#created Regression model to make the predictions.
from sklearn.linear_model import LogisticRegression
Model=LogisticRegression()
Model.fit(X,Y)

predictions=Model.predict(X)

#Imported matrix model to get the accuracy.
from sklearn.metrics import confusion_matrix
confusion_matrix(Y,predictions)

#Accuracy of my model is 88%
from sklearn.metrics import accuracy_score
accuracy_score(Y,predictions)

#88% of the client has subscribed to TERM DEPOSIT.







