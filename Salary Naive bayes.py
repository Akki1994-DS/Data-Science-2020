# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:15:12 2020

@author: axays
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

Salary_Train=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\SalaryData_Train (1).csv')
Salary_Test=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\SalaryData_Test (1).csv')

Salary_Train.columns



#Created dummy variables for categorical values into binary format for Salary_Train data.
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

Salary_Train.workclass=le.fit_transform(Salary_Train.workclass)
Salary_Train.education=le.fit_transform(Salary_Train.education)
Salary_Train.maritalstatus=le.fit_transform(Salary_Train.maritalstatus)
Salary_Train.occupation=le.fit_transform(Salary_Train.occupation)
Salary_Train.relationship=le.fit_transform(Salary_Train.relationship)
Salary_Train.race=le.fit_transform(Salary_Train.race)
Salary_Train.sex=le.fit_transform(Salary_Train.sex)
Salary_Train.native=le.fit_transform(Salary_Train.native)
Salary_Train.Salary=le.fit_transform(Salary_Train.Salary)

#Created dummy variables for categorical values into binary format for Salary_Test data.
Salary_Test.workclass=le.fit_transform(Salary_Test.workclass)
Salary_Test.education=le.fit_transform(Salary_Test.education)
Salary_Test.maritalstatus=le.fit_transform(Salary_Test.maritalstatus)
Salary_Test.occupation=le.fit_transform(Salary_Test.occupation)
Salary_Test.relationship=le.fit_transform(Salary_Test.relationship)
Salary_Test.race=le.fit_transform(Salary_Test.race)
Salary_Test.sex=le.fit_transform(Salary_Test.sex)
Salary_Test.native=le.fit_transform(Salary_Test.native)
Salary_Test.Salary=le.fit_transform(Salary_Test.Salary)

#Defined input output variables.
Train_X=Salary_Train.iloc[:,0:13]
Train_Y=Salary_Train.iloc[:,13]
Test_X=Salary_Test.iloc[:,0:13]
Test_Y=Salary_Test.iloc[:,13]


ignb=GaussianNB()

pred_gausaaian=ignb.fit(Train_X,Train_Y)
predictions=ignb.predict(Test_X)

pred_gausaaian1=ignb.fit(Train_X,Train_Y)
predictions2=ignb.predict(Train_X)

confusion_matrix(Train_Y,predictions2)

from sklearn.metrics import accuracy_score
accuracy_score(Train_Y,predictions2)







