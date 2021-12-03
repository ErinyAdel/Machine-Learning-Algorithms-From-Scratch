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
df_train_valid, df_train, df_valid, df_test = model.splitDataset(df)

### Preprocessing
x_train, y_train = model.preprocessFeatures(df_train)
x_valid, y_valid = model.preprocessFeatures(df_valid)
x_test, y_test  = model.preprocessFeatures(df_test)

### Create Model For Training Data
logRegModel = model.trainLogisticReg(x_train, y_train,x_valid, y_valid) 

### Evaluation
tp, tn, fp, fn = model.convusionMatrix(y_test, x_valid, logRegModel)

cm = np.array([
            [tp, fp], 
            [fn, tn]
])
print(cm, '\n')
print((cm / cm.sum()).round(3), '\n') 

precision = tp / (tp + fp)   
print('Precision:', precision, '\n')

recall = tp / (tp + fn)   
print('Recall:', recall)
print('Model Faild To Classify', format((1-recall)*100, '.2f'), '% Correctly\n')



### Saving The Model
import pickle 


with open('churn-model.bin', 'wb') as f_out: ## wb: Write in Binary File
    pickle.dump(logRegModel, f_out)
print("Congrats! .. The Model Saved.\n")

### Loading The Model
with open('churn-model.bin', 'rb') as f_in: ## rb: Read in Binary File
    loadedModel = pickle.load(f_in)
    #print(loadedModel)

# customer = {
#     'customerid': '8879-zkjof',
#     'gender': 'female',
#     'seniorcitizen': 0,
#     'partner': 'no',
#     'dependents': 'no',
#     'tenure': 41,
#     'phoneservice': 'yes',
#     'multiplelines': 'no',
#     'internetservice': 'dsl',
#     'onlinesecurity': 'yes',
#     'onlinebackup': 'no',
#     'deviceprotection': 'yes',
#     'techsupport': 'yes',
#     'streamingtv': 'yes',
#     'streamingmovies': 'yes',
#     'contract': 'one_year',
#     'paperlessbilling': 'yes',
#     'paymentmethod': 'bank_transfer_(automatic)',
#     'monthlycharges': 79.85,
#     'totalcharges': 3320.75
# }

# import requests
# url = 'http://localhost:9696/predict'
# response = requests.post(url, json=customer)
# result = response.json()
# print(result)




