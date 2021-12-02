# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:09:09 2021
@author: Eriny
"""

import pandas as pd
import numpy as np

import LR_Model as model

import warnings
warnings.simplefilter("ignore")

import seaborn as sns
from matplotlib import pyplot as plt

### Loading Data
df = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
#print(df)
df.columns = df.columns.str.lower().str.replace(' ', '_')

### Splittig The Dataset To Training, Validation and Testing Sets (80%, 20%, 20%)
df_train_full, x_train, y_train, x_valid, y_valid, x_test, y_test = model.splitDataset(df)


### Preprocessing
X = model.preprocessFeatures(df_train_full)

### Training logistic regression



















