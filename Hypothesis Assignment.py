import pandas as pd
import numpy as np
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

Cutlets=pd.read_csv('C:\\Users\\axays\\Desktop\\Data Sets\\Cutlets.csv')

#Normality test
print(stats.shapiro(Cutlets['Unit A']))
print(stats.shapiro(Cutlets['Unit B']))

#P-values for Unit A is 0.31 and Unit B is 0.52 which is greater than 0.05. It means data is following normal disrtibution.

scipy.stats.levene(Cutlets['Unit A'],Cutlets['Unit B'])

#2 Sample T test
sm.stats.ttest_ind(Cutlets['Unit A'],Cutlets['Unit B'])

#================================================================

LabTAT=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\LabTAT.csv')
LabTAT.columns=['Laboratory1','Laboratory2','Laboratory3','Laboratory4']



#Normality test
print(stats.shapiro(LabTAT['Laboratory1']))
print(stats.shapiro(LabTAT['Laboratory2']))
print(stats.shapiro(LabTAT['Laboratory3']))
print(stats.shapiro(LabTAT['Laboratory4']))

#P-values for all Laboratories is  greater than 0.05. It means data is following normal disrtibution.

scipy.stats.levene(LabTAT['Laboratory1'],LabTAT['Laboratory2'],LabTAT['Laboratory3'],LabTAT['Laboratory4'])

#One Way Anova Test

MOD=ols('Laboratory1~Laboratory2+Laboratory3+Laboratory4',data=LabTAT).fit()
AOV_Table=sm.stats.anova_lm(MOD, Type=1)
print(AOV_Table)

#====================================================================

Ratio=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\BuyerRatio.csv')

#Total number of sales of product for Male (N1=393)
#Total number of sales of product for Female (N2=4064)
N1=393
P1=0.08
393/4457

N2=4064
P2=0.91
4064/4457

Population1=np.random.binomial(1,P1,N1)
Population2=np.random.binomial(1,P2,N2)

#2 Proportion Test
sm.stats.ttest_ind(Population1, Population2)

sm.stats.ttest_ind(Population1, Population2,alternative='smaller')

#=================================================================

#For this data i am not able to get the cross table because of multiple variables need suggestion for same.
Customer=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\Costomer+OrderForm.csv')

pd.pivot_table(Customer)
count
x=pd.crosstab(Customer['Malta'],Customer['India'])
x

#==================================================================

Faltoons=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\Faltoons.csv')
X=pd.crosstab(Faltoons['Weekdays'], Faltoons['Weekend'])

#Total number of males on weekdays+weekend and probability.
G1=167
H1=0.42
167/400

#Total number of females on weekdays+weekend and probability.
G2=233
H2=0.58
233/400

Population1=np.random.binomial(1,H1,G1)
Population2=np.random.binomial(1,H2,G2)

#2 Proportion Test
sm.stats.ttest_ind(Population1, Population2)
sm.stats.ttest_ind(Population1, Population2,alternative='smaller')






















