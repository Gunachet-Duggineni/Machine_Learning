#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#reading the dataset
dataSet = pd.read_csv('Machine_learning\Regression\Data4.csv')
X = dataSet.iloc[:, 1:-1].values
Y = dataSet.iloc[:, -1].values

#Training in polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
X_Poly = poly_regressor.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X_Poly, Y)


#Visualising the polynomial regression model results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color= 'red')
plt.plot(X_grid, lin_regressor.predict(poly_regressor.fit_transform(X_grid)), color= 'blue')
plt.title('Linear regression result')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


print(lin_regressor.predict(poly_regressor.fit_transform([[6.5]])))