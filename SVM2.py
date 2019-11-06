# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:24:42 2019

@author: Shivam-PC
"""


# importing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
X = dataset.iloc[:, 1:-2].values
y = dataset.iloc[:, -1].values

#cleaning data
c=[0,2,3,5,15]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
for i in c:
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features = [2])

c=[6,7,8,9,10,11,12,13,14]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
for i in c:
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features = [3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

X[:, 16] = labelencoder_X.fit_transform(X[:, 16])
onehotencoder = OneHotEncoder(categorical_features = [4])
    
    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

y= labelencoder_X.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features = [2])


#spitting the dataset into training set and text set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.15, random_state=42)

# Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
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