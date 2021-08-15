# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:15:15 2020

@author: axays
"""


import pandas as pd
import numpy as np

Forest=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\forestfires.csv')

Forest.head()
Forest.describe()
Forest.columns
Forest.isnull().sum()
Forest.nunique()
Forest['size_category'].unique()
Forest.dtypes

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

Forest.month=le.fit_transform(Forest.month)
Forest.day=le.fit_transform(Forest.day)
Forest.size_category=le.fit_transform(Forest.size_category)

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

# Made original wine data normally distributed.
Forest_norm=norm_func(Forest.iloc[:,0:])
Forest_norm.describe()


from sklearn.model_selection import train_test_split
train,test=train_test_split(Forest_norm, test_size=0.2)


#Defined Salary column as output and rest as input variable.
Train_X=Forest_norm.iloc[:,0:30]
Train_Y=Forest_norm.iloc[:,30]
Test_X=Forest_norm.iloc[:,0:30]
Test_Y=Forest_norm.iloc[:,30]

#Created SVM model and got the Accuracy of 74.85%.
from sklearn.svm import SVC
model_linear=SVC(kernel='linear')
model_linear.fit(Train_X,Train_Y)
Prediction=model_linear.predict(Test_X)
np.mean(Prediction==Test_Y)

#kernal = poly
model_linear=SVC(kernel='poly')
model_linear.fit(Train_X,Train_Y)
Prediction=model_linear.predict(Test_X)
np.mean(Prediction==Test_Y)

#kernal = rbf
model_linear=SVC(kernel='rbf')
model_linear.fit(Train_X,Train_Y)
Prediction=model_linear.predict(Test_X)
np.mean(Prediction==Test_Y)







