# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:32:44 2020

@author: axays
"""

import pandas as pd
import numpy as np

Concrete=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\concrete.csv')

Concrete.head()
Concrete.columns
Concrete.isnull().sum()
Concrete.dtypes

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

Concrete_norm=norm_func(Concrete.iloc[:,0:])
Concrete_norm.describe()
Concrete_norm.shape

from keras.models import Sequential
from keras.layers import Dense

Model=Sequential()
Model.add(Dense(50,input_dim=8,activation='relu'))
Model.add(Dense(40,activation='relu'))
Model.add(Dense(20,activation='relu'))
Model.add(Dense(1,kernel_initializer='normal'))
Model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse'])

Column_names=list(Concrete_norm.columns)
Predictors=Column_names[0:8]
Target=Column_names[8]

First_Model=Model
First_Model.fit(np.array(Concrete_norm[Predictors]),np.array(Concrete_norm[Target]),epochs=10)
Pred_train=First_Model.predict(np.array(Concrete_norm[Predictors]))
Pred_train=pd.Series([i[0]for i in Pred_train])
RMSE=np.sqrt(np.mean((Pred_train-Concrete_norm[Target])**2))
