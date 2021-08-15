# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:23:29 2020

@author: axays
"""


import pandas as pd
import numpy as np
import seaborn as sns
Glass=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\glass.csv')

#EDA

Glass.columns
Glass.isnull().sum()
Glass.describe()
Glass.nunique()
Glass['Type'].unique()

#The below is the list of Type of glass
#1 building_windows_float_processed
#2 building_windows_non_float_processed
#3 vehicle_windows_float_processed
#4 vehicle_windows_non_float_processed (none in this database)
#5 containers
#6 tableware
#7 headlamps




#Training and Testing using 
from sklearn.model_selection import train_test_split
train,test=train_test_split(Glass, test_size=0.3)

#KNN using sklearn
from sklearn.neighbors import KNeighborsClassifier as KNC

#for 3 Nearest Neighbors
Neighbors=KNC(n_neighbors=3)

#Here i have selected Type as output and rest is input.
Neighbors.fit(train.iloc[:,0:8], train.iloc[:,9])

#Checking trainig accuracy.
Train_acc=np.mean(Neighbors.predict(train.iloc[:,0:8])==train.iloc[:,9]) #85% Accuracy

Test_acc=np.mean(Neighbors.predict(test.iloc[:,0:8])==test.iloc[:,9]) #63% accuracy

#for 5 Nearest Neighbors
Neighbors=KNC(n_neighbors=5)

Neighbors.fit(train.iloc[:,0:8], train.iloc[:,9])

#Checking trainig accuracy.
Train_acc1=np.mean(Neighbors.predict(train.iloc[:,0:8])==train.iloc[:,9]) #76% Accuracy

Test_acc1=np.mean(Neighbors.predict(test.iloc[:,0:8])==test.iloc[:,9]) #66% accuracy




