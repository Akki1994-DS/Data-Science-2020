# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 19:54:06 2020

@author: axays
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

Emp=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\emp_data.csv')

Emp.head()

plt.hist(Emp.Salary_hike)
plt.hist(Emp.Churn_out_rate)

sns.countplot(x='Salary_hike',hue='Churn_out_rate',data=Emp)

plt.plot(Emp.Salary_hike, Emp.Churn_out_rate,'ro');plt.xlabel('Salary_hike');plt.ylabel('Churn_out_rate')

Emp.corr()
#Relation between two variable is strong.

Model=smf.ols('Churn_out_rate~Salary_hike',data=Emp).fit()

type(Model)

Model.summary()

Model.conf_int(0.05)
predictions=Model.predict(Emp)
