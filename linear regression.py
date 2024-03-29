# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:10:18 2019

@author: Shivam-PC
"""

# importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('kc_house_data.csv')
X = dataset.iloc[:,5].values
y = dataset.iloc[:, 2].values
co = dataset.iloc[:, :]
x2=[]

for i in X:
    x2.append([i])
X=x2
#finding co relation of dataset
corel=co.corr()
plt.matshow(co.corr())
plt.show()


#splitting dataset
#spitting the dataset into training set and text set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)


# Fitting data into simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test,y_test))

#prediction
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs squreft (Training set)')
plt.ylabel('Price')
plt.xlabel('Squareft')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs squreft (Training set)')
plt.ylabel('Price')
plt.xlabel('Squareft')
plt.show()