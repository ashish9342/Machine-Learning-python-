import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
#integer location based from 0 to length-1
# takes all values except last column from [:,:-1]

# instead of removing the data,
# take the mean of data for missing value
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

print(X)
print('\n\n')



# dependent = X.iloc[:,3].values
#integer location based from 0 to length-1
# [:,3] removes values from 0 to 3

# print(dependent)
