# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:53:43 2021
@author: Eriny
"""

''' Dimensional reduction (Compress the image) '''

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

## Select random points from given data (X). (KxN) 
def _init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k,n))
    idx = np.random.randint(0, m, k) ## Random 3 nums from 0:300(m) 
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]

    return centroids

## Selection step (in K-Means algorithm). Centroid function
def _find_closest_centroids(X, centroids):
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
def _compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)  
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

## K-Means function
def k_means(X, k, max_iters):
    m, n = X.shape
    idx = np.zeros(m)
    centroids = _init_centroids(X, k)               ## Init. centroids
    
    for i in range(max_iters):
        idx = _find_closest_centroids(X, centroids) ## Selection step
        centroids = _compute_centroids(X, idx, k)   ## Update centroids step
    
    return idx, centroids


### 
k = 16

### Load the image data  
image_data = loadmat('./bird_small.mat')
print(image_data)
A = image_data['A']
print(A.shape) # (128, 128, 3) --> 128x128 --> 16384 Features
plt.imshow(A)
plt.show()

### Preprocessing for data
## Normalize value ranges
A = A / 255.

## Reshape the array
X = np.reshape(A, (A.shape[0]*A.shape[1], A.shape[2]))
print(X.shape)


### K-Means algorithm
idx, centroids = k_means(X, k, 10)

## map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]

## Reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

plt.imshow(X_recovered)
plt.show()