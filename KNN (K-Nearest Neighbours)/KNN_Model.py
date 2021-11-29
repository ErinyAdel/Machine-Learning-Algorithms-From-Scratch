# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 11:07:10 2021
@author: Eriny
"""

import numpy as np
from collections import Counter 

## Global Function                                             ______________
def EuclideanDistance(x1, x2):        ##(General Case)    d = √ Σⁿ (qi - pi)²
    return np.sqrt(np.sum(x1-x2)**2)  ##                       ⁱ⁼⁰                     
                                                            

class KNNModel:
    
    def __init__(self, k = 3):
        self.k = k
        
    ## Fit The Training Set Samples        
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    ## Predict New Samples
    def predict(self, new_x):
        predicted_labels = [self._predict(x) for x in new_x]
        return np.array(predicted_labels)
        
    ## Private Method
    def _predict(self, x):
        ## Compute Distances 
        distances = [EuclideanDistance(x, xTrain) for xTrain in self.x_train]
        ## Get (K) Nearest Neighbours/Samples, Labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self.y_train[i] for i in k_indices]
        ## Majority Vote, Most Common Class Label
        most_common = Counter(k_nearest).most_common(1)
        
        return most_common[0][0] ## Index of the most common value.
    
    
"""
a = [1, 1, 1, 2, 2, 3, 4, 5, 6]
com = Counter(a).most_common(3)
print(com[0][0])
"""
