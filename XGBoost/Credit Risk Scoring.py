# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 09:32:07 2021
@author: Eriny
"""

''' Models Are Dependently Trained --> Sequential '''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

### Load The Data
df = pd.read_csv('CreditScoring.csv')
df.columns = df.columns.str.lower()

### Preprocessing

#print(df.status.value_counts())
status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}
df.status = df.status.map(status_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}
df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}
df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}
df.job = df.job.map(job_values)


#print(df.describe().round().T)
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)


## Split The Data
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_valid = train_test_split(df_train_full, test_size=0.25, random_state=11)

y_train = (df_train.status == 'default').values ## 1: default, 0 otherwise
y_valid = (df_valid.status == 'default').values

del df_train['status']
del df_valid['status']

df_train_dict = df_train.fillna(0).to_dict(orient='records')
df_valid_dict = df_valid.fillna(0).to_dict(orient='records')

## 
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train_dict)
X_valid = dv.transform(df_valid_dict)

### Model
