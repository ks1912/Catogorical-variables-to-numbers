# DATA PREPROCESSING
# Importing the libraries 
--------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
--------------------------------------------------------------------------
# IMPORTING DATA SET'S

dataset = pd.read_csv('editeddata.csv')
--------------------------------------------------------------------------
# Creating Veriable Vector's

# Creating independent variable vector

X = dataset.iloc[:,:-1].values
# print(X) # Uncommnet X to print X

# Creating dependent variable vector

Y = dataset.iloc[:,3].values 
# print(Y) Uncomment Y to print Y
--------------------------------------------------------------------------
# Finding mean and updating it in the dataset

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN",strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
--------------------------------------------------------------------------
# Changing Catogorical data from lavels to labels and unbiasing them

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableencoder_X = LabelEncoder()
X[:, 0] = lableencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

lableencoder_Y = LabelEncoder()
Y = lableencoder_Y.fit_transform(Y)


