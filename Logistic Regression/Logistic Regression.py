# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:06:37 2021
@author: Eriny
"""

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #
"""
Logistic Regression Algorithm:

    Need to probability not to contious values for the output.
    So we use "Sigmoid Function" in the 'Linear Model'
• Approximation
    
• Line Equation:
                f(w,b) = wx + b                 ----> Contious Output Values
                                 1
                ŷ = hϴ(x) = ـــــــــــــــــــــــــــــــ        ----> Discrete Output Values
                            1 + e⁻ʷˣ﹢ᵇ
            
• Sigmoid Function:             1
                    s(x) = ـــــــــــــــــــــــــ   , z = ŷ = h(ϴ)           s(x) >= 0.5 --> 1, 
                            1 + e⁻ᶻ                               s(x) < 0.5 --> 0

• Cost Function:
    Cross-Entropy = J(w,b) = 1/N Σⁿ [yⁱ.log(hϴ(xⁱ)) + (1 - yⁱ).log(1 - hϴ(xⁱ))]
                                ⁱ⁼¹
                           = 1/N Σⁿ (yⁱ - (w.xⁱ + b)²
                                ⁱ⁼¹
• Gradient Descent: To minimize the error/cost function.
     Derivative of J = J'(w,b) = J'(ϴ) =  ______________________________________
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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
## Import My Model
from LogisticRegression_Model import LogisticRegression


## 
LEARNING_RATE = 0.0001
EPOCHS = 1000

## Model Accuracy For Testing Data 
def accuracy(y_true, y_predicted):
    return np.sum(y_true == y_predicted) / len(y_true)
    


data_set = datasets.load_breast_cancer()
x, y     = data_set.data, data_set.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

logReg_model = LogisticRegression(lr=LEARNING_RATE, n_iters=EPOCHS)
logReg_model.fit(x_train, y_train)
predictions = logReg_model.predict(x_test)

print("Logistic Regression classification Accuracy:", accuracy(y_test, predictions))



