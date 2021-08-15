# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:56:04 2020

@author: axays
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time

Plastic=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\PlasticSales.csv')

Plastic.head()
Plastic.isnull().sum()
Plastic.dtypes
Plastic.nunique()

# Converting the normal index of Amtrak to time stamp 
Plastic.index = pd.to_datetime(Plastic.Month,format="%b-%y")

colnames = Plastic.columns
colnames

Plastic["Sales"].plot() # time series plot

# Creating a Date column to store the actual Date format for the given Month column
Plastic["Date"] = pd.to_datetime(Plastic.Month,format="%b-%y")


# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

Plastic["month"] = Plastic.Date.dt.strftime("%b") # month extraction
#Amtrak["Day"] = Amtrak.Date.dt.strftime("%d") # Day extraction
#Amtrak["wkday"] = Amtrak.Date.dt.strftime("%A") # weekday extraction
Plastic["year"] =Plastic.Date.dt.strftime("%Y") # year extraction

# Boxplot for ever
sns.boxplot(x="Month",y="Sales",data=Plastic)
sns.boxplot(x="year",y="Sales",data=Plastic)

# Line plot for Ridership based on year  and for each month
sns.lineplot(x="year",y="Sales",hue="Month",data=Plastic)

# moving average for the time series to understand better about the trend character in plastic
Plastic.Sales.plot(label="org")
for i in range(2,24,6):
    Plastic["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(Plastic.Sales,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Plastic.Sales,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(Plastic.Sales,lags=10)
tsa_plots.plot_pacf(Plastic.Sales)

Train = Plastic.head(48)
Test = Plastic.tail(12)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales)

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales)

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)



























