# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 20:54:44 2021
@author: Eriny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

learningRate         = 0.00001
regularizationDegree = 5

def plot_inputs(x1, x2):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x1['Test 1'], x1['Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(x2['Test 1'], x2['Test 2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Accepted')
    ax.set_ylabel('Rejected')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, lr):
    ''' '''
    
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    y_0_term = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    y_1_term  = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    
    reg = (lr / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    
    return np.sum(y_0_term - y_1_term ) / (len(X)) + reg



def gradient(theta, X, y, learningRate):
    ''' '''
    
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] =(np.sum(term)/len(X))+((learningRate/len(X))*theta[:,i])
    
    return grad



def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]



### Load The Data
data = pd.read_csv('data.txt', header=None, names=['Test 1', 'Test 2', 'Admitted'])

accepted = data[ data['Admitted'].isin([1]) ]
rejected = data[ data['Admitted'].isin([0]) ]

plot_inputs(accepted, rejected)



x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'x0', 1)   # adding x0

'''
x1 + x1^2 + x1x2 + x1^3 + x1^2 x2 + x1 x2^2 + x1^4 + x1^3 x2 + x1^2 x2^2 + x1 x2^3


F10 = x1

F20 = x1^2
F21 = x1 x2

F30 = x1^3
F31 = x1^2 x2
F32 = x1 x2^2

F40 = x1^4
F41 = x1^3 x2
F42 = x1^2 x2^2
F43 = x1 x2^3 

'''
for i in range(1, regularizationDegree): # 1,2,3,4
    for j in range(0, i):  # 0 , 1 , 2 ,2
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j) # i=3 , j=2

data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)

print('data \n' , data.head(10))

print('................................................')



# set X and y (remember from above that we moved the label to column 0)
cols = data.shape[1]
print('cols = ' , cols)
print('................................................')

X2 = data.iloc[:,1:cols]
print('X2 = ')
print(X2.head(10))
print('................................................')

y2 = data.iloc[:,0:1]
print('y2 = ')
print(y2.head(10))
print('................................................')

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(X2.shape[1])

print('theta 2 = ' , theta2)
print('................................................')


rcost = cost(theta2, X2, y2, learningRate)
print()
print('regularized cost = ' , rcost)
print()


## Optimization
result = opt.fmin_tnc(func=cost, x0=theta2, fprime=gradient, args=(X2, y2, learningRate))

new_thetas = result[0]

new_thetas = np.matrix(new_thetas)
predictions = predict(new_thetas, X2)

correct = [1 if a == b else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('Accuracy = {0}%'.format(accuracy))

