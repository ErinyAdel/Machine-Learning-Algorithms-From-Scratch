from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression


### Splittig The Dataset To Training, Validation and Testing Sets (80%, 20%, 20%)
def splitDataset(df):
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_valid = train_test_split(df_train_full, test_size=0.25, random_state=11) # 20/80
            
    return df_train_full, df_train, df_valid, df_test


###
def preprocessFeatures(df):
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
    
    ## Normalization
    #global_churn_rate = df.churn.value_counts(normalize=True).round(2) ## 1 (Churn): 0.27 Churn Rate, ...
    #global_churn_rate = df.churn.mean().round(2)  
    global_churn_rate = (df.churn.sum() / df.churn.count()).round(2)
    #print(global_churn_rate)
    
    
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
        df_group = df.groupby(by=c).churn.agg(['count', 'mean'])
        df_group['difference'] = df_group['mean'] - global_churn_rate
        df_group['risk_ratio'] = df_group['mean'] / global_churn_rate
        print( df_group, '\n')
    
    ## Feature Importance: Mutual Information --> (Tell us how important each categorical variable is).
    ## The Higher MI is, The More We Learn About The Chart By Observing The Value.
    def calcMI(series): ## labels_true, labels_pred
        return mutual_info_score(series, df.churn)
    
    df_mi = df[categorical].apply(calcMI)
    df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
    print(df_mi, '\n')
    
    
    ## Feature Importance: Correlation --> (Tell us how important each numerical variable is).
    ## Positive Correlation: ↑ A Var & ↑ Another. Negative Correlation: ↑ A Var & ↓ Another. 
    ## Measures Dependency Between Two Variables --> -1 <= c <= 1
    df_cc = df[numerical].corrwith(df.churn).to_frame('correlation')       ## Know Direction
    print(df_cc, '\n')
    df_cc = df[numerical].corrwith(df.churn).abs().to_frame('correlation') ## Know Importance
    print(df_cc, '\n')
    
    #print( df_train_full.groupby(by='churn')[numerical].mean() )
    
    
    ## One-Hot Encoding Categorical Features
    #features = categorical.copy()
    features = []
    category_values = {}
    for c in categorical + numerical:
        category_values[c] = list(df[c].value_counts().head().index)
        #print(category_values[c])
    for c, values in category_values.items():
        for v in values:
            feature = '%s_%s' % (c, v)
            #print(feature)
            df[feature] = (df[c] == v).astype('int')
            features.append(feature)       
    X = df[features].values
    #print(df[features].columns)
    #print(X[0])
    #print(df.iloc[0])
    
    return X, df['churn']

def logisticRegression(X, y, C=1.0):
    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    #print( model.coef_[0].round(3) )      ## Weights For 1st Row
    #print( model.intercept_[0].round(3) ) ## Bias For 1st Row
        
    return model


def predict(X, model):
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

def trainLogisticReg(x_train, y_train, x_valid, y_valid):
    C = 1.0
    n_splits = 5
            
    for i in range(n_splits):        
        model    = logisticRegression(x_train, y_train, C=C)
        y_pred   = predict(x_valid, model)
        modelAcc = accuracy(y_pred, y_valid)
    
    print('Validation Accuracy:', format((modelAcc*100), '.2f'), '%\n')

    df_pred = pd.DataFrame()
    df_pred['Probability'] = y_pred
    df_pred['Prediction']  = modelAcc.astype(int)
    df_pred['Actual']      = y_valid.values
    df_pred['IsCorrect']   = df_pred.Prediction == df_pred.Actual
    print(df_pred)
    
    return model

def accuracy(y_pred, y_valid):
    ## Threshold Value: 0.5 is The Best Value (Try From 0:1)
    churn_pred_result = (y_pred >= 0.5) 
    accuracy = (y_valid == churn_pred_result).mean()
    return accuracy

def convusionMatrix(y_test, x_valid, model):
    '''
        ## Confusion Matrix: TP, TN, FP, FN
        ## Accuracy: True Positive / Total -- TP / N --> The Mean -- Afected By Class Imbalance
        ## Precision: Fraction of Positive Predictions Are Correct -- TP / (TP + FP)
        ## Recall:    Fraction of Correctly Identified Positive    -- TP / (TP + FN)
    '''
    y_pred = predict(x_valid, model)
    
    actual_positive = (y_test == 1)
    actual_negative = (y_test == 0)
    predicted_positive = (y_pred >= 0.5) 
    predicted_negative = (y_pred < 0.5)
    
    tp = (predicted_positive == actual_positive).sum()
    tn = (predicted_negative == actual_negative).sum()
    fp = (predicted_positive != actual_positive).sum()
    fn = (predicted_negative != actual_negative).sum()
    
    return tp, tn, fp, fn