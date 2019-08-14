# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:11:34 2019

@author: Shivam-PC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))


y_pred = regressor.predict(X_test)

print('Intercept \n',regressor.intercept_)
print('Coefficient \n', regressor.coef_)

