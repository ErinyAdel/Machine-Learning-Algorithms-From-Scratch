# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:09:09 2021
@author: Eriny
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import LR_Model as model

import warnings
warnings.simplefilter("ignore")

### 


### Loading Data
df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
#print(df)
df.columns = df.columns.str.lower().str.replace(' ', '_')


### Splittig The Dataset To Training, Validation and Testing Sets (80%, 20%, 20%)
#df_train, df_val, df_test, y_train, y_val, y_test = model.splitDataset(df)
df_train, df_valid, df_test = model.splitDataset(df)

### Preprocessing
x_train, y_train = model.preprocessFeatures(df_train)
x_valid, y_valid = model.preprocessFeatures(df_valid)


### Create Model For Training Data
## Create
logRegModel = LogisticRegression(solver='liblinear', random_state=1)

## Train
logRegModel.fit(x_train, y_train)

## Get Predictions For Evaluation
predictions = logRegModel.predict(x_train)
#print(predictions)
y_pred = logRegModel.predict_proba(x_train)[:, 1]
predicted_result = (y_pred >= 0.5)
#print(predicted_result)
accuracy = (y_train == predicted_result).mean()
print('Training Accuracy:', format((accuracy*100), '.2f'), '%')

#print( logRegModel.coef_[0].round(3) )      ## Weights For 1st Row
#print( logRegModel.intercept_[0].round(3) ) ## Bias For 1st Row

logRegModel = LogisticRegression(solver='liblinear', random_state=1)
logRegModel.fit(x_valid, y_valid)
predictions = logRegModel.predict(x_train)
y_pred = logRegModel.predict_proba(x_valid)[:, 1]
y_pred = (y_pred >= 0.5)
accuracy = (y_valid == y_pred).mean()
print('Validation Accuracy:', format((accuracy*100), '.2f'), '%')



'''
[ 0.563 -0.086 -0.599 -0.03  -0.092  0.1   -0.116 -0.106 -0.027 -0.095
 -0.323  0.317 -0.116  0.001 -0.168  0.127 -0.081  0.136 -0.116 -0.142
  0.258 -0.116 -0.264 -0.213  0.091 -0.048 -0.074 -0.027 -0.136  0.175
 -0.134  0.127 -0.249  0.297 -0.085 -0.116  0.079 -0.099 -0.116  0.093
  0.178 -0.116 -0.184 -0.069  0.   ]
-0.122


[-0.203 -0.12  -0.147 -0.176 -0.097 -0.226 -0.264 -0.059 -0.202 -0.062
 -0.059  0.39  -0.484 -0.228  0.196 -0.291 -0.228  0.108 -0.202 -0.228
  0.021 -0.116 -0.228  0.106 -0.201 -0.228 -0.17   0.075 -0.228 -0.168
  0.074 -0.228  0.923 -1.131 -0.115 -0.015 -0.307  0.174 -0.188 -0.121
 -0.188  1.697 -0.891  1.083  0.54   0.762  0.737  0.468  0.104  0.068
  0.171 -0.192 -0.131]
-0.323



'''






