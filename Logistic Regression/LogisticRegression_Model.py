"""
Created on Sat Oct 23 14:06:56 2021
@author: Eriny
"""

import numpy as np

class LogisticRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr      = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias    = None
        
    def fit(self, x, y):
        ## Init. Parameters
        n_samples, n_features = x.shape
        self.weights          = np.zeros(n_features)
        self.bias             = 0 ## Or Random Numbers
        
        ## Gradient Descent
        for _ in range(self.n_iters): 
            linear_model = np.dot(x, self.weights) + self.bias ## f(w,b) = wx + b 
            y_predicted  = self._sigmoid(linear_model)         ## 1 / (1 + e⁻ᶻ)
                        
            dw = 1/n_samples * np.dot(x.T, (y_predicted-y) ) ## = dw = 1/N Σ 2xⁱ(ŷ - yⁱ)
            db = 1/n_samples * np.sum(y_predicted - y)       ## = dw = 1/N Σ 2(ŷ - yⁱ)
            
            self.weights -=  self.lr * dw ## w = w - α.dw
            self.bias    -=  self.lr * db ## b = b - α.db
    
    def predict(self, x):
        linear_model   = np.dot(x, self.weights) + self.bias ## f(w,b) = wx + b 
        y_predicted    = self._sigmoid(linear_model)         ## s(x)   = 1 / (1 + e⁻ᶻ) 
        predictedClass = [1 if i > 0.5 else 0 for i in y_predicted] 
        return predictedClass

    def _sigmoid(self, new_x):
        return 1 / (1 + np.exp(-new_x))