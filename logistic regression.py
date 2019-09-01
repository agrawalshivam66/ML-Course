# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:06:48 2019

@author: Shivam-PC
"""

# importing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Admission_Predict.csv")
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, -1].values

for i in range(len(y)):
    if y[i]>=0.6:
        y[i]=1
    else:
        y[i]=0

#spitting the dataset into training set and text set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)

# Fitting Logistic regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(classifier.score(X_test,y_test))
print('Confusion matrix:')
print(cm)
tn, fp, fn, tp = cm.ravel()

precision=tp/(tp+fp)
recall=tp/(tp+fn)

print('precison-',precision)
print('recall-',recall)