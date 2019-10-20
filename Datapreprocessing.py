# -*- coding: utf-8 -*-
# DATA PREPROCESSING
# Importing the libraries 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# IMPORTING DATA SET'S
datasetnew = pd.read_csv('Data.csv')
X = datasetnew.iloc[:,:-1].values # Creating independent variable vector
# print(X)
Y = datasetnew.iloc[:,3].values # Creating dependent variable vector
# print(Y)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "nan",strategy = 'mean',axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableencoder_X = LabelEncoder()
X[:, 0] = lableencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

lableencoder_Y = LabelEncoder()
Y = lableencoder_Y.fit_transform(Y)


