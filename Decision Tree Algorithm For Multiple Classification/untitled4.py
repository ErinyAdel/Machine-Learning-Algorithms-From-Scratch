# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 15:18:41 2021
@author: Eriny
"""

import pandas as pd
import numpy as np
import DT_Model as model

### Load The Data
df = pd.read_csv('CreditScoring.csv')
df.columns = df.columns.str.lower()
print(df)



