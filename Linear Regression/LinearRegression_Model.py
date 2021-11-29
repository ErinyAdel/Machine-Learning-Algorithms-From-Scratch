# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 12:53:18 2021
@author: Eriny
"""

import numpy as np

class LinearRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr      = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias    = None
        
    def fit(self, x, y):
        ## Init. Parameters
        n_samples, n_features = x.shape
        self.weights          = np.zeros(n_features)
        self.bias             = 0
        
        ## Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(x, self.weights) + self.bias ## ŷ = wx + b 
            
            dw = 1/n_samples * np.dot(x.T, (y_predicted-y) ) ## = dw = 1/N Σ 2xⁱ(ŷ - yⁱ)
            db = 1/n_samples * np.sum(y_predicted - y)       ## = dw = 1/N Σ 2(ŷ - yⁱ)
            
            self.weights -=  self.lr * dw ## w = w - α.dw
            self.bias    -=  self.lr * db ## b = b - α.db
    
    def predict(self, x):
        y_predicted = np.dot(x, self.weights) + self.bias ## ŷ = wx + b 
        return y_predicted
