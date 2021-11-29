from sklearn.model_selection import train_test_split
import numpy as np


### Splittig The Dataset To Training, Validation and Testing Sets (80%, 20%, 20%)
def splitDataset(df):
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_valid = train_test_split(df_train_full, test_size=0.25, random_state=11) # 20/80
    
    ### Splitting The Datasets To Xs, Y
    y_train = df_train.churn.values
    y_valid = df_valid.churn.values
    y_test  = df_test.churn.values
    
    del df_train['churn']
    del df_valid['churn']
    del df_test['churn']
    
    x_train = df_train
    x_valid = df_valid
    x_test  = df_test
    
    return df_train_full, x_train, y_train, x_valid, y_valid, x_test, y_test