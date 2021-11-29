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
print(df.dtypes)
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



## Exploratory Data Analysis
#print(df.churn.value_counts(normalize=True)) # 1 (Churn): 0.26537 Churn Rate = df.churn.mean(), ... 
global_churn_rate = round(df.churn.mean(), 2)

## Replacing Missing Values By 0s

## Feature Importance Mutual Information
def calculate_mi(series):
    return mutual_info_score(series, df_train_full.churn)


categorical = list(df.dtypes[df.dtypes == 'object'].index)
numerical = ['tenure', 'monthlycharges', 'totalcharges']

df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
#print(df_mi)

## Feature importance
female_mean = df_train_full[df_train_full.gender == 'female'].churn.mean()
print('gender == female:', round(female_mean, 3))

male_mean = df_train_full[df_train_full.gender == 'male'].churn.mean()
print('gender == male:  ', round(male_mean, 3))

partner_yes = df_train_full[df_train_full.partner == 'yes'].churn.mean()
print('partner == yes:', round(partner_yes, 3))

partner_no = df_train_full[df_train_full.partner == 'no'].churn.mean()
print('partner == no :', round(partner_no, 3))

print(female_mean / global_churn_rate)
print(male_mean / global_churn_rate)
print(partner_yes / global_churn_rate)
print(partner_no / global_churn_rate)


df_group = df_train_full.groupby(by='gender').churn.agg(['mean'])
df_group['diff'] = df_group['mean'] - global_churn_rate
df_group['risk'] = df_group['mean'] / global_churn_rate
print( df_group.columns )





