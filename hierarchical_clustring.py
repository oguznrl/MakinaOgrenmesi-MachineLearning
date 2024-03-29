import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 

X1,y1=make_blobs(n_samples=50,centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
plt.scatter(X1[:, 0], X1[:, 1], marker='o') 

agglom=AgglomerativeClustering(n_clusters=4,linkage="average")

agglom.fit(X1,y1)

plt.figure(figsize=(6,4))

x_min,x_max=np.min(X1,axis=0),np.max(X1,axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    colors=plt.cm.nipy_spectral(agglom.labels_[i] / 10.)
    print(agglom.labels_[i])
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=colors,
             fontdict={'weight': 'bold', 'size': 9})
#Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])

# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()

dist_matrix = distance_matrix(X1,X1) 
print(dist_matrix)
Z = hierarchy.linkage(dist_matrix, 'average')
dendro = hierarchy.dendrogram(Z)