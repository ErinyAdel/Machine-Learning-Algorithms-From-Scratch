"""
Created on Sat Oct 23 09:20:05 2021
@author: Eriny
"""

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #
"""
KNN (K-Nearest Neighbours) Algorithm:

• Claculating The Distance Of The New Sample To Each K Samples Of The Training Set.
  Then Classify It To The Most Nearest Neighbours' Class/Group.

• Euclidean Distance:      _______________________
      (2D Case)      d = √ (X2 - X1)² + (Y2 - Y1)²
                     
• Euclidean Distance:      _____________
   (General Case)    d = √ Σⁿ (qi - pi)²
                          ⁱ⁼⁰
"""
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

## Import My KNN Model (In Same Directory)
from KNN_Model import KNNModel         


iris_dataset = datasets.load_iris()
x, y = iris_dataset.data, iris_dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

## Understanding The Dataset

## N-D Matrix
#print(x_train.shape) ## (120, 4) ==> 120 Samples, 4 Features (Columns)
#print(x_train[0])     ## Get First Sample (First Raw Of Samples)

## 1D Vector
#print(y_train.shape) ## (120,) ==> 120 Samples (Output)
#print(y_train)       ## Get The 120 Samples

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) ## Red, Green, Blue (3 Features)
plt.figure()                                             ## Two Classes & 2D Feature Vector ???
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()


knn = KNNModel(k = 5)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)

