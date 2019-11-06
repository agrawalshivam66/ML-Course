# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:24:42 2019

@author: Shivam-PC
"""

# importing 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target


#spitting the dataset into training set and text set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X , y, test_size=0.2, random_state=0)

# Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, C=10)
classifier.fit(X_train, y_train)

# predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
print('hard margin')
cm1 = confusion_matrix(y_test, y_pred)
print(classifier.score(X_test,y_test))
print(cm1)


# Fitting SVM to training set
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'linear', random_state = 0, C=0.01)
classifier2.fit(X_train, y_train)


# predicting the results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
print('soft margin')
cm1 = confusion_matrix(y_test, y_pred)
print(classifier2.score(X_test,y_test))
print(cm1)

h=0.2

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
for i in range(1):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    fig = plt.gcf()
    fig.set_size_inches(12,10)
    plt.title('hard margin')

plt.show()

h=0.2

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
for i in range(1):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = classifier2.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('soft margin')
    fig = plt.gcf()
    fig.set_size_inches(12,10)

plt.show()

