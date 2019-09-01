# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:20:13 2019

@author: Shivam-PC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 6.0)

#Preprocessing Input data
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:,0].values
y = data.iloc[:, 1].values

#spitting the dataset into training set and text set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)


# Fitting data into simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
#print(regressor.score(X_test,y_test))

#prediction
y_pred = regressor.predict(X_test.reshape(-1,1))

# Visualising the Test set results
plt.scatter(X_train, y_train, color = 'blue')
regression_line = plt.plot(X_train, regressor.predict(X_train.reshape(-1,1)), color = 'red')


#Performing gradient descent on the data
#Building the model
m = 0
c = 0

L = 0.01
epochs = 100

n = float(len(X))

for i in range(epochs):
  Y_pred = m*X + c
  D_m = (-2/n) * sum(X * (y - Y_pred))
  D_c = (-2/n) * sum(y - Y_pred)
  m = m - L * D_m
  c = c - L * D_c

print(m,c)


#Making predictions
Y_pred = m*X + c
plt.scatter(X,y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='yellow')
plt.show()