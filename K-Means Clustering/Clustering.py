# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:15:48 2021
@author: Eriny
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


## Select random points from given data (X). (KxN) 
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k,n))
    idx = np.random.randint(0, m, k) ## Random 3 nums from 0:300(m) 
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]

    return centroids

## Selection step (in K-Means algorithm). Centroid function
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m) ## Zeros vector for each data row. To cluster each row to its cluster number (0, 1, 2)
   
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2) ## (X - each centroid)²
            if dist < min_dist:
                min_dist = dist
                idx[i] = j ## Cluster each row to group (G0, G1, G2)
    return idx

## Update centroid points
def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)  
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

## K-Means function
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids) ## Selection step
        centroids = compute_centroids(X, idx, k)   ## Update centroids step
    
    return idx, centroids

###
k = 3 ## Number of clusters

### Load data
data = loadmat('.\ex7data2.mat')
print(data)
#print(data['X'])
print(data['X'].shape) ## m=300, n=2

### Apply K-Means

## Classify points --> Random & Initial centers KxN --> K: Clusters num., N: Features num.
X = data['X']
initial_centroids = np.array([[8, 0], [8, 6], [0, 3]]) # Initial
#initial_centroids = init_centroids(X, k)                # Random
print('Initial Centroids:\n',initial_centroids)
 
## 1st Cluster/Classify X
idx = find_closest_centroids(X, initial_centroids)
#print(idx)
 
## 2nd Update centroids
updatedCentroids = compute_centroids(X, idx, k)
print('Updated Centroids:\n',updatedCentroids)

## Prepeat 1st & 2nd steps # times --> 5+4+3+2+1 times.
for x in range(6):
    idx, centroids = run_k_means(X, initial_centroids, x)
    print('Updated Centroids:\n',centroids)
    
    ## Draw it
    cluster1 = X[np.where(idx == 0)[0],:]
    cluster2 = X[np.where(idx == 1)[0],:]
    cluster3 = X[np.where(idx == 2)[0],:]
    
    fig, ax = plt.subplots(figsize=(9,6))
    ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
    ax.scatter(centroids[0,0],centroids[0,1],s=300, color='r')
    
    ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
    ax.scatter(centroids[1,0],centroids[1,1],s=300, color='g')
    
    ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
    ax.scatter(centroids[2,0],centroids[2,1],s=300, color='b')
    
    ax.legend()

  
