# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:44:21 2020

@author: axays
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

Calories=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\calories_consumed.csv')

Calories.columns=['Weightgained','CaloriesConsumed']

plt.hist(Calories.Weightgained)
plt.hist(Calories.CaloriesConsumed)
plt.plot(Calories.CaloriesConsumed, Calories.Weightgained,'ro');plt.xlabel('Weightgained');plt.xlabel('CaloriesConsumed')
Calories.corr()

Model=smf.ols('Weightgained~CaloriesConsumed',data=Calories).fit()
type(Model)

Model.params

Model.summary()


Model.conf_int(0.05)
predictions=Model.predict(Calories)

p.corr(Calories.Weightgained)

plt.scatter(x=Calories['CaloriesConsumed'],y=Calories['Weightgained'],color='blue');plt.plot(Calories['Weightgained'],predictions,color='black');plt.xlabel('CALORIES');plt.ylabel('WEIGHT')         
