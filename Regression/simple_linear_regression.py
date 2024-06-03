#Machine_learning/Data2.csv

#importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataSet = pd.read_csv('Machine_learning\Regression\Data2.csv')
X = dataSet.iloc[:, :-1].values
Y = dataSet.iloc[:, -1].values

#Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Training the simple regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#Predicting the test, train set results
Y_test_pred = regressor.predict(X_test)
Y_train_pred = regressor.predict(X_train)


#Visualising the training set results
plt.scatter(X_train, Y_train, color= 'red')
#Creates the scatter plot of the given inputs of the total values

#Visualising the training set results
plt.plot(X_train, Y_train_pred, color= 'blue') #Regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
 

#visualising the test set results
plt.scatter(X_test, Y_test, color= 'red')
plt.plot(X_test, Y_test_pred, color= 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
