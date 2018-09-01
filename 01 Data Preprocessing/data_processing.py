import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

independent = dataset.iloc[:,:-1].values
#integer location based from 0 to length-1
# takes all values except last column from [:,:-1]


print(independent)
print('\n\n')



dependent = dataset.iloc[:,3].values
#integer location based from 0 to length-1
# [:,3] removes values from 0 to 3
print(dependent)
