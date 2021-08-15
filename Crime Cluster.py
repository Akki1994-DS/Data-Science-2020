# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:09:03 2020

@author: axays
"""


import pandas as pd
import matplotlib.pyplot as plt

Crime=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\crime_data.csv')

#EDA
Crime.isnull().sum()
Crime.columns
Crime.shape
Crime.describe()
Crime.nunique()

#Data is normally distributed because min value is 0 and max is 1.
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

Crime_norm=norm_func(Crime.iloc[:,1:])

Crime_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

#Using linkage function found out the distances
Dist=linkage(Crime_norm,method='complete',metric='euclidean')


#Created a dendrogram
plt.figure(figsize=(15,5));plt.title('Hierarchical Clustering Dendogram');plt.xlabel('index');plt.ylabel('Distance')
sch.dendrogram(Dist,leaf_rotation=0.,leaf_font_size=8.,)
plt.show()

#Creating AgglomerativeClustering for chossing as 4 clusters.
from sklearn.cluster import AgglomerativeClustering
F_Complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(Crime_norm)
F_Complete.labels_

cluster_labels=pd.Series(F_Complete.labels_)

#Adding cluster column to the dataset.
Crime['Cluster']=cluster_labels
Crime=Crime.iloc[:,[5,0,1,2,3,4]]
Crime.head()

#getting aggregate mean of each cluster.
Group=Crime.groupby(Crime.Cluster).mean()
Group.head()

# Cluster 0 is consist of highest crime where i can mark it as #Tier 1
# Cluster 3 is at second highest position in crime which is #Tier 2
# Cluster 1 has less crime record which is #Tier 3
# Lowest crime occur is in cluster 2 where i can mark it as #Tier 4