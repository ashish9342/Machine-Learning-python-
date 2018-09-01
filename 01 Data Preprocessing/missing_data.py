import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values # independent variable
Y = dataset.iloc[:,3].values # dependent variable

#integer location based from 0 to length-1
# takes all values except last column from [:,:-1]

# instead of removing the data,
# take the mean of data for missing value
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)
print('\n\n')

#strategy mean, median, most_frequent
# mean : average of numbers 1, 10 = 5.5
# median : (1,10) center no. 5 and 6, if two no. take the average of them

#-------------------------------------
# Encoding categorial data (country and purchased)
# for machine to understand, or put them in equation
# encodes country

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#---------------------------------------------------------
#  Splitting the Dataset into the Training set and Test  (20% to 40%)
#  cross_validation => model_selection
#  ML can adapt the new model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )

#-----------------------------------------------------------------
# Feature Scaling
# standardization and normalization
# Putting the variables in same range and same scale, so that no variable dominates by other
# -1 < value < 1
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


