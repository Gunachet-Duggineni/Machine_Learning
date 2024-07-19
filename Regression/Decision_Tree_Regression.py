"""Decision tree regression observes features of an object and
trains a model in the structure of a tree to predict data in the future
to produce meaningful continuous output."""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataSet = pd.read_csv('Machine_learning\Regression\Data4.csv')
X = dataSet.iloc[:, 1:-1].values
Y = dataSet.iloc[:, -1].values

#Training the model on the datasets
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()
DTR.fit(X, Y)

#Predicting a single feature
print(DTR.predict([[6.5]]))

#Visualising the datasets
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color= 'red')
plt.plot(X_grid, DTR.predict(X_grid), color= 'blue')
plt.title('Decision tree regression result')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

