# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:09:09 2021
@author: Eriny
"""

import pandas as pd
from sklearn.metrics import mutual_info_score


import LR_Model as model
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

### Loading Data
df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
#print(df)
df.columns = df.columns.str.lower().str.replace(' ', '_')


### Preprocessing
#print(df.dtypes)
## 
categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_cols:
    df[c] = df[c].str.lower().str.replace(' ', '_')

## 
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce').isnull()
#print(df['totalcharges'].sum())
df['totalcharges'] = df['totalcharges'].fillna(0)
#print(df['totalcharges'].isnull().sum())

## Convert Y Column (Churn) From Categorical Type To Numerical Type (0, 1)
df['churn'] = (df['churn'] == 'yes').astype(int)
#print(df.churn)


### Splittig The Dataset To Training, Validation and Testing Sets (80%, 20%, 20%)
df_train_full, x_train, y_train, x_valid, y_valid, x_test, y_test = model.splitDataset(df)



### Exploratory Data Analysis 
## Normalization
#global_churn_rate = df.churn.value_counts(normalize=True).round(2) ## 1 (Churn): 0.27 Churn Rate, ...
#global_churn_rate = df.churn.mean().round(2)  
global_churn_rate = (df.churn.sum() / df.churn.count()).round(2)
print(global_churn_rate)


categorical = list(df.dtypes[df.dtypes == 'object'].index)
del categorical[0] ## customerid
numerical = ['tenure', 'monthlycharges', 'totalcharges']
#print(categorical)

## Feature importance

#print(df.partner.value_counts(normalize=True))
#partner_yes = df_train_full[df_train_full.partner == 'yes'].churn.mean()
#partner_no = df_train_full[df_train_full.partner == 'no'].churn.mean()

## Difference: Global - Group --> Low Difference ( < 0) With High Ratio --> High Importance
#print(global_churn_rate - partner_yes)
#print(global_churn_rate- partner_no)
## Risk Ratio: Group / Global --> High Risk ( > 1) With High Ratio --> High Importance
#print(partner_yes / global_churn_rate)
#print(partner_no / global_churn_rate)

for c in categorical:
    df_group = df_train_full.groupby(by=c).churn.agg(['count', 'mean'])
    df_group['difference'] = df_group['mean'] - global_churn_rate
    df_group['risk_ratio'] = df_group['mean'] / global_churn_rate
    print( df_group, '\n')

## Feature Importance: Mutual Information --> (Tell us how important each categorical variable is).
## The Higher MI is, The More We Learn About The Chart By Observing The Value.
def calcMI(series): ## labels_true, labels_pred
    return mutual_info_score(series, df_train_full.churn)

df_mi = df_train_full[categorical].apply(calcMI)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
print(df_mi, '\n')


## Feature Importance: Correlation --> (Tell us how important each numerical variable is).
## Positive Correlation: ↑ A Var & ↑ Another. Negative Correlation: ↑ A Var & ↓ Another. 
## Measures Dependency Between Two Variables --> -1 <= c <= 1
df_cc = df_train_full[numerical].corrwith(df_train_full.churn).to_frame('correlation')       ## Know Direction
print(df_cc, '\n')
df_cc = df_train_full[numerical].corrwith(df_train_full.churn).abs().to_frame('correlation') ## Know Importance
print(df_cc, '\n')

#print( df_train_full.groupby(by='churn')[numerical].mean() )


## One-hot encoding














