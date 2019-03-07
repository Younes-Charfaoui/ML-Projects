import pandas as pd
import numpy as np
df = pd.read_csv('customers.csv')

# Preprocessing
df.drop(['Region', 'Channel'], axis = 1, inplace = True)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler().fit(df)
min_max = MinMaxScaler().fit(df)
stand_data = scaler.transform(df)
minmax_data = min_max.transform(df)
log_data = np.log(df)

from sklearn.decomposition import PCA
reduced_log = PCA(2).fit_transform(log_data)
reduced_stand = PCA(2).fit_transform(stand_data)
reduced_minmax = PCA(2).fit_transform(minmax_data)

import matplotlib.pyplot as plt
plt.scatter(reduced_log[:,0] , reduced_log[:,1])
plt.show()
plt.scatter(reduced_stand[:,0] , reduced_stand[:,1])
plt.show()
plt.scatter(reduced_minmax[:,0] , reduced_minmax[:,1])
plt.show()

# CLustering
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters = 2).fit(reduced_log)
kmeans_result = kmeans.predict(reduced_log)
kmeans_score = silhouette_score(reduced_log ,kmeans_result)
print(' The score for Kmeans is ' , kmeans_score)

dbscan = DBSCAN().fit(reduced_log)
dbscan_result = dbscan.fit_predict(reduced_log)
dbscan_score = silhouette_score(reduced_log ,dbscan_result)
print(' The score for DBscan is ' , dbscan_score)


ms = MeanShift().fit(reduced_log)
ms_result = ms.predict(reduced_log)
ms_score = silhouette_score(reduced_log ,ms_result)
print(' The score for MeanShift is ' , ms_score)

ac = AgglomerativeClustering().fit(reduced_log)
ac_result = ac.fit_predict(reduced_log)
ac_score = silhouette_score(reduced_log ,ac_result)
print(' The score for AgglomerativeClustering is ' , ac_score)

gmm = GaussianMixture(n_components=2).fit(reduced_log)
gmm_result = gmm.predict(reduced_log)
gmm_score = silhouette_score(reduced_log ,gmm_result)
print(' The score for GaussianMixture is ' , gmm_score)

# Best score of the Kmeans









