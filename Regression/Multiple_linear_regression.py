#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#reading the dataset
dataSet = pd.read_csv('Machine_learning\Regression\Data3.csv')
X = dataSet.iloc[:, :-1].values
Y = dataSet.iloc[:, -1].values


#encoding the categorical variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))



#splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Training the multiple linear regression model using the given training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the results
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)

#concatenate expects the input in a tuple "()" form
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

