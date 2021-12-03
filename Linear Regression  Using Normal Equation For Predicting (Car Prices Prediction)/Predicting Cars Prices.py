# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 10:58:16 2021
@author: Eriny
"""

''' Singular Matrix: A matrix whose determinant is 0 and hence it has no inverse. 
                     Has 2 >= columns with same values
'''

import pandas as pd
import numpy as np

import LR_Model as model

df = pd.read_csv('./data.csv')
#print( len(df) )
#print( df.columns )

## Data Preparation 
df.columns = df.columns.str.lower().str.replace(' ', '_')
#print( df.columns )

#print( df.dtypes )
strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Validation framework
n = len(df)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

# Suffle the indices
np.random.seed(2)
idx = np.arange(n) 
np.random.shuffle(idx)
df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val   = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test  = df_shuffled.iloc[n_train+n_val:].copy()

#
y_train_orig = df_train.msrp.values
y_val_orig   = df_val.msrp.values
y_test_orig  = df_test.msrp.values

y_train = np.log1p(df_train.msrp.values)
y_val   = np.log1p(df_val.msrp.values)
y_test  = np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']


x_train = model.featuresPreprocessing(df_train)
x_val = model.featuresPreprocessing(df_val)
x_test = model.featuresPreprocessing(df_test)

w_0, w = model.train_linear_regression(x_train, y_train)
y_pred = w_0 + x_train.dot(w)
error = model.rmse(y_train, y_pred)
print("Training Error: ", error)

y_pred = w_0 + x_val.dot(w)
val_error = model.rmse(y_val, y_pred)
print("Validation Error: ", val_error)


## Using the model
x_train_valid = np.concatenate([x_train, x_val])
y_train_valid = np.concatenate([y_train, y_val])

w_0, w = model.train_linear_regression(x_train_valid, y_train_valid)
y_pred = w_0 + x_test.dot(w)
error = model.rmse(y_test, y_pred)
print("Testing Error: ", error)
print(x_train_valid.shape)
print(y_train_valid.shape)


'''
without age
Training Error:  0.7554192603920133
Validation Error:  0.7616530991301607

with age
Training Error:  0.5175055465840046
Validation Error:  0.5172055461058338

with num of doors
Training Error:  0.5150615580371418
Validation Error:  0.515799564150262

with top 5 make
Training Error:  0.5058876515487503
Validation Error:  0.50760388495572

without top 5 & with all categorical hot encoded
Training Error:  767.9798498917206
Validation Error:  797.4129620273194

with all categorical hot encodied & regularization
Training Error:  0.45430642661056736
Validation Error:  0.48230567255041834

'''