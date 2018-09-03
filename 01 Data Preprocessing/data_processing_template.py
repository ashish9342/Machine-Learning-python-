"""
Created on Sat Sep  1 22:36:12 2018

@author: Ashish
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values # independent variable
Y = dataset.iloc[:,3].values # dependent variable

print(X)
#---------------------------------------------------------
#  Splitting the Dataset into the Training set and Tes
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )

#-----------------------------------------------------------------
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""