# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:24:52 2020

@author: axays
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.api as sm

Toyota=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\ToyotaCorolla.csv',encoding='latin1')

#EDA

Toyota1=Toyota.drop(['Id', 'Model', 'Mfg_Month', 'Mfg_Year','Fuel_Type', 'Met_Color', 'Color', 'Automatic','Cylinders', 'Mfr_Guarantee','BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2','Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player','Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio','Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim','Radio_cassette','Tow_Bar'],axis=1)

Toyota1.columns

Toyota1.head()

Toyota1.shape

Toyota1.describe()


Toyota1.nunique()
Toyota1['Age_08_04'].unique()

Toyota1.isnull().sum()

x=Toyota1.corr()

sns.pairplot(Toyota1)

# As per the EDA part the relation between all the variables are <0.85.
# however when you see the pairplot there is no relation between the variables.
