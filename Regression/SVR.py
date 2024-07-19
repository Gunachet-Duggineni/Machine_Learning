"""This method uses kernerls, i.e a kernel is a special function that can be used to convert a 2d model iinto a 3d plane."""


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataSet = pd.read_csv('Machine_learning\Regression\Data4.csv')
X = dataSet.iloc[:, 1:-1].values
Y = dataSet.iloc[:, -1].values

#feature scaling is applied here
#As standard scaler expects a 2d array, we have to reshape Y from1D to 2D
Y = Y.reshape(len(Y), 1)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


#Scaling of the features matrix and the dependent variable vectore should be different
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

#Training teh SVR model on the data set
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(X, Y)

#Predicting with SVR model but you have to reverse the scaling to get an actual result in the "Y" matrix
Y_predicted = sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

#Visualising the SVR result
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color= 'red')
plt.plot(X_grid, sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color= 'blue')
plt.title('Support vector regression result')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()