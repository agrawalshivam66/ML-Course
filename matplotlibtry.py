# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:11:04 2019

@author: Shivam-PC
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv('kc_house_data.csv')
X = dataset.iloc[:50, 2].values



plt.scatter( X,y)
plt.xlabel('price')
plt.ylabel('square feet')
plt.title("housing data")
plt.legend()
plt.show()

plt.plot( X,y)
plt.xlabel('price')
plt.ylabel('square feet')
plt.title("housing data")
plt.legend()
plt.show()

