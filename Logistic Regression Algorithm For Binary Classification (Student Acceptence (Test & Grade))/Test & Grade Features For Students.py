# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 07:52:04 2021
@author: Eriny
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def plot_inputs(x1, x2):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x1['Test'], x1['Grade'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(x2['Test'], x2['Grade'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Accepted')
    ax.set_ylabel('Rejected')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    '''          ₘ 
    J() = -1/m [ Σ  y⁽ᶦ⁾.log h₍Θ₎(x⁽ᶦ⁾) + (1 - y⁽ᶦ⁾).log(1 - h₍Θ₎(x⁽ᶦ⁾))]
                ᶦ⁼¹
    '''  
    y_0_term = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    y_1_term = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    
    return np.sum(y_0_term - y_1_term) / (len(X))

def gradient(theta, X, y):
    ''' For Thetas '''
    theta = np.matrix(theta)
    parameters = int(theta.ravel().shape[1]) ## 3
    grad = np.zeros(parameters)              ## [0. 0. 0.]
    
    error = sigmoid(X * theta.T) - y         ## ŷ - y --> Predicted - True
    
    for i in range(parameters):
        term    = np.multiply(error, X[:,i]) 
        grad[i] = np.sum(term) / len(X)
    
    return grad


def predict(theta, X):
    ''' One-Hot Encoding The Output (Probabilities)'''
    probability = sigmoid(X * theta.T)
    return [1 if prob >= 0.5 else 0 for prob in probability]

### Load The Data
data = pd.read_csv('data.txt', header=None, names=['Test', 'Grade', 'Admitted'])

accepted = data[ data['Admitted'].isin([1]) ]
rejected = data[ data['Admitted'].isin([0]) ]

plot_inputs(accepted, rejected)

## add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'X0', 1)

NO_OF_FEATURES = 3

## Initialize Theta Parameter As Array 1x3 (3 Features --> Ones, Test, Grade)
theta = np.zeros(NO_OF_FEATURES)

## Split The Data
cols = data.shape[1]
X = data.iloc[:, :cols-1] 
y = data.iloc[:, cols-1:]

## Convert X & y & theta To Matrices & Vectors
X     = np.array(X.values)
y     = np.array(y.values)
theta = np.array(theta) #Same


print('\nError Before Taining:', format(cost(theta, X, y), '.2f'))

## Optimization
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
#print(result) ## OUTPUT: (array([4.67051226e-05, 3.06592387e-03, 3.09290654e-03]), 7, 0) ([theta], try_no, 0)
new_thetas = result[0]

print('\nError After Taining:', format(cost(new_thetas, X, y), '.2f'))


new_thetas = np.matrix(new_thetas)
predictions = predict(new_thetas, X)

correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('\nModel Accuracy = {0}%'.format(accuracy))
