# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:44:29 2020

@author: NARAYANA REDDY DATASCIENTIST
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# READ THE DATASET

dataset=pd.read_csv('50_Startups.csv')

# DIVIDE THE DATA SET INTO X AND Y
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

# LABEL ENCODING
states=pd.get_dummies(x['State'],drop_first=True)

# DROP THE STATE COLUMN
x=x.drop('State',axis=1)

# CONCAT DUMMY VARIABLE
x=pd.concat([x,states],axis=1)

# SPLITTING THE DATASET INTO TRAINING AND TEST DATASET

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# FITTING MULTIPLE LINER REGRESSION MODEL TO TRAINING DATASET

from sklearn.linear_model import LinearRegression
linearregression=LinearRegression()
linearregression.fit(x_train,y_train)

# PREDICT THE MODEL
y_predict=linearregression.predict(x_test)

# MODEL PERFORMANCE SCORE

from sklearn.metrics import r2_score
r2score=r2_score(y_test,y_predict)
