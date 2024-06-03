#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset
dataSet = pd.read_csv('Machine_learning/Data1.csv')
X = dataSet.iloc[:, :-1].values
Y = dataSet.iloc[:, -1].values  #Assuming that the dependent variable vector is in the last column



#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')

imputer.fit(X[:, 1:3])      
X[:, 1:3] = imputer.transform(X[:, 1:3])



#Encoding categorical data in the features matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


#Encoding categorical data in the dependent vector matrix
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


#Splitting the data into train and test sets 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
#The 1st and 3rd elements get the 80% of the data here.
#So the training sets must be at the 1, 3 positions


from sklearn.preprocessing import StandardScaler #Feature scaling after splitting the training and test sets
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
#Only transform is used as the same scaler has to be used on the test set instead of a different one

print(X_train)