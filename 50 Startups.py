# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:06:06 2020

@author: axays
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.api as sm

Startups=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\50_Startups.csv')

# EDA
Startups.head()

Startups.shape

Startups.describe()

Startups.columns

Startups.nunique()
Startups['State'].unique()

Startups.isnull().sum()

Startups.corr()

sns.pairplot(Startups)

Startups.columns=['RandDSpend', 'Administration', 'MarketingSpend', 'State', 'Profit']
Startups1=Startups.drop(['State'],axis=1)
Startups1.columns

# REGRESSION MODEL EVALUATION
Model=smf.ols('Profit~RandDSpend+Administration+MarketingSpend',data=Startups1).fit()
type(Model)

Model.summary()

Model_1=smf.ols('Profit~Administration',data=Startups1).fit()

Model_1.summary()

Model_2=smf.ols('Profit~MarketingSpend',data=Startups1).fit()

Model_2.summary()

Model_3=smf.ols('Profit~Administration+MarketingSpend',data=Startups1).fit()

Model_3.summary()

#iNDIVIDUALLY MY BOTH MODEL_1 AND MODEL_2 PERFORMING WELL HOWEVER WHEN COMBINING TOGETHER MODEL_3 IS NOT PERFORMING WELL DUE TO PVALUES.

#INFLUENCE INDEX PLOT
sm.graphics.influence_plot(Model)

#Dropping the influence values which is 48 and 49.
Startups_New=Startups1.drop(Startups1.index[[48,49]],axis=0)

#Creating a new model after removing influncing values.
Model_New=smf.ols('Profit~RandDSpend+Administration+MarketingSpend',data=Startups_New).fit()

Model_New.summary()

#Added variable plot
sm.graphics.plot_partregress_grid(Model_New)

#HOWEVER AS PER VARIABLE PLOT MY NEW MODEL IS BEST FIT MODEL.

Startups_Predictions=Model_New.predict(Startups_New)

