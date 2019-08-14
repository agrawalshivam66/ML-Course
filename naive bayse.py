# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:12:54 2019

@author: Shivam-PC
"""

# importing 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :]  
y = iris.target


#spitting the dataset into training set and text set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)

# Fitting Naive Bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
print('GaussianNB-',end=' ')
print(classifier.score(X_test,y_test))


# Fitting Naive Bayes to training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm2= confusion_matrix(y_test, y_pred)
print('MultinomialNB-',end=' ')
print(classifier.score(X_test,y_test))


# Fitting Naive Bayes to training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB(binarize=0.5)
classifier.fit(X_train, y_train)

# predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred)
print('BernoulliNB-',end=' ')
print(classifier.score(X_test,y_test))
print('Gaussian NB confusion matrix-')
print(cm1)
