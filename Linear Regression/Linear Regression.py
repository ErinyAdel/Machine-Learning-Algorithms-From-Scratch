"""
Created on Fri Oct 22 23:33:30 2021
@author: Eriny
"""

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #
"""
Linear Regression Algorithm:
    
• Approximation
    
• Line Equation:
                ŷ = wx + b 
    w: Weights (Slope)               -----> How do we find
    x: 
    b: Bias (Shift along y-axis)     -----> these values ?

• Cost Function:
    MSE (Mean Square Error) = J(w,b) = 1/N Σⁿ (yⁱ - ŷⁱ)²
                                          ⁱ⁼¹
                                     = 1/N Σⁿ (yⁱ - (w.xⁱ + b)²
                                          ⁱ⁼¹
• Gradient Descent: To minimize the error/cost function.
     Derivative of J = J'(w,b) = ______________________________________
                                | df/dw = 1/N  Σⁿ 2xⁱ(yⁱ - (w.xⁱ + b))|
                                |             ⁱ⁼¹                     |
                                | df/db = 1/N  Σⁿ 2(yⁱ - (w.xⁱ + b))  |
                                |_____________ⁱ⁼¹ ____________________|

     ∟ Gradient Descent: Iterative Method To Get The Minimum
         Updated Rules:
                         w = w - α.dw
                         b = b - α.db
                    
         dj/dw = dw = 1/N Σ -2xⁱ(yⁱ - (w.xⁱ + b))
                    = 1/N Σ -2xⁱ(yⁱ - ŷ)
                    = 1/N Σ 2xⁱ(ŷ - yⁱ)

         dj/db = db = 1/N Σ -2(yⁱ - (w.xⁱ + b))
                    = 1/N Σ -2(yⁱ - ŷ)
                    = 1/N Σ 2(ŷ - yⁱ)


• Learning Rate:
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
## Import My Model
from LinearRegression_Model import LinearRegression


## Mean Square Error
def MSE(y_true, y_predicted):
    return np.mean( (y_true - y_predicted)**2 )


x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

## 
#plt.figure(figsize=(8,6))
#plt.scatter(x[:, 0], y, color='b', marker='o', edgecolors='k', s=30)
#plt.show()

## Understanding The Dataset
## N-D Matrix
#print(x_train.shape) ## (80, 1) ==> 80 Samples, 1 Feature (Column)
#print(x_train[0])     ## Get First Sample (First Raw Of Samples) -- Between 0 To 4 ==> random_state=4
## 1D Vector
#print(y_train.shape) ## (80,) ==> 80 Samples (Output)
#print(y_train)       ## Get The 80 Samples

reg_model = LinearRegression(lr=0.01)

reg_model.fit(x_train, y_train)
predictions = reg_model.predict(x_test)
error = MSE(y_test, predictions)
print("Testing Error: " + str(error))
cmap = plt.get_cmap('viridis')
m1 = plt.scatter(x_train, y_train, color=cmap(0.5), s=10)
m2 = plt.scatter(x_test, y_test, color=cmap(1.2), s=10)
plt.plot(x_test, predictions, color='black', linewidth=2 ,label="Predictions")
plt.show()


predictions = reg_model.predict(x)
error = MSE(y, predictions)
print("Dataset Error: ", error)
m1 = plt.scatter(x_train, y_train, color='blue', s=10)
m2 = plt.scatter(x_test, y_test, color='red', s=10)
plt.plot(x, predictions, color='black', linewidth=2 ,label="Predictions")
plt.show()