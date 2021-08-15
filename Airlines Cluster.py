# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:33:49 2020

@author: axays
"""

import pandas as pd
import matplotlib.pyplot as plt

Airlines=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\EastWestAirlines.csv')

#EDA
Airlines.isnull().sum()
Airlines.columns
Airlines.shape
Airlines.describe()
Airlines.nunique()
Airlines['cc1_miles'].unique()
Airlines1=Airlines.drop(['ID#'],axis=1)

#Normalizing the data to minimize the distance.
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

Airlines1_norm=norm_func(Airlines.iloc[:,1:])

Airlines1_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

#Using linkage function found out the distances
Dist=linkage(Airlines1_norm,method='complete',metric='euclidean')

#Created a dendrogram
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendogram');plt.xlabel('index');plt.ylabel('Distance')
sch.dendrogram(Dist,leaf_rotation=0.,leaf_font_size=8.,)
plt.show()

#Creating AgglomerativeClustering for chossing as 5 clusters.
from sklearn.cluster import AgglomerativeClustering
G_Complete=AgglomerativeClustering(n_clusters=5,linkage='complete',affinity='euclidean').fit(Airlines1_norm)
G_Complete.labels_

cluster_labels=pd.Series(G_Complete.labels_)

#Adding cluster column to the dataset.
Airlines1['Cluster']=cluster_labels
Airlines1=Airlines1.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
Airlines1.head()

#getting aggregate mean of each cluster.
Group=Airlines1.groupby(Airlines1.Cluster).mean()






