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

logRegModel = LogisticRegression(solver='liblinear', random_state=1) 
logRegModel.fit(x_train, y_train) 

#print( logRegModel.coef_[0].round(3) )      ## Weights For 1st Row
#print( logRegModel.intercept_[0].round(3) ) ## Bias For 1st Row

y_pred = logRegModel.predict_proba(x_valid)[:, 1]
accuracy = model.accuracy(y_pred, y_valid)

print('Validation Accuracy:', format((accuracy[1]*100), '.2f'), '%\n')

df_pred = pd.DataFrame()
df_pred['Probability'] = y_pred
df_pred['Prediction']  = accuracy[0].astype(int)
df_pred['Actual']      = y_valid.values
df_pred['IsCorrect']   = df_pred.Prediction == df_pred.Actual
print(df_pred)


### Evaluation

## Confusion Matrix: TP, TN, FP, FN
## Accuracy: True Positive / Total -- TP / N --> The Mean -- Afected By Class Imbalance
## Precision:
## Recall:    

actual_positive = (y_valid == 1)
actual_negative = (y_valid == 0)
predicted_positive = (y_pred >= 0.5) 
predicted_negative = (y_pred < 0.5)

tp = (predicted_positive == actual_positive).sum()
tn = (predicted_negative == actual_negative).sum()
fp = (predicted_positive != actual_positive).sum()
fn = (predicted_negative != actual_negative).sum()

cm = np.array([
            [tp, fp], 
            [fn, tn]
])
print(cm)
print((cm / cm.sum()).round(3)) 

    














