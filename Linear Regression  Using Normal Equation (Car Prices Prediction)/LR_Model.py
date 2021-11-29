import numpy as np

# Baseline solution --> Use Numerical Features
#print( df.dtypes )
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
#print( df[base].values , '\n\n')
#print( df_train[base].values , '\n\n')

def featuresPreprocessing(df):
    df = df.copy()
    features = base.copy()
    
    ## Get Age From The Given Year Value
    max_year = df.year.max() 
    df['age'] = max_year - df.year
    features.append('age') # features = features + ['age']
    
    ## One-Hot Encoding For Categorical Features
    for v in range(2, int(df.number_of_doors.max()+1)):
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
    ## One-Hot Encoding For Categorical Features    
    """
    makes = list(df.make.value_counts().head().index) ## Take Top 5 
    for v in makes:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)        
    """
    categorices = list(df.dtypes[df.dtypes == 'object'].index)
    category_values = {}
    for c in categorices:
        category_values[c] = list(df[c].value_counts().head().index)
    for c, values in category_values.items():
        for v in values:
            feature = '%s_%s' % (c, v)
            df[feature] = (df[c] == v).astype('int')
            features.append(feature)            
    ##            
    X = df[features].fillna(0).values
    return X


# Linear Regression --> Normal Equation with Regularization
def train_linear_regression(X, y, r=0.001): ## Loop over [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10] to get best r value
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X]) ## Add X₀ = 1 for each row data
    
    XTX = X.T.dot(X)
    ## Regularization
    ## ____________________________
    
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    ## ____________________________
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)    ## w = (XᵀX)⁻¹ Xᵀy [The Normal Equation]
    
    return w[0], w[1:]


def rmse(y, y_pred):
    error = y_pred - y        #  _____________
    mse = (error ** 2).mean() # √1/m Σ(ŷ - y)²
    return np.sqrt(mse)