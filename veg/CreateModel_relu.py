# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:47:46 2019

@author: Ranajoy
"""
import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


advt = pd.read_csv( "DemandDataSet_Pencil_Beans.csv" )

advt = advt[["v","p","r","h","pr","se","st","ev","we","ve","lo","su","qt","pr1","su1","wed","fe","na","cl","ex","Quantity"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  advt[["v","p","r","h","pr","se","st","ev","we","ve","lo","su","qt","pr1","su1","wed","fe","na","cl","ex"]],
  advt.Quantity,
  test_size=0.3,
  random_state = 42 )
  
#Count features for modelization
X_num_columns= len(X_train.columns)

#Define model
model = Sequential()

model.add(Dense(300,
                activation='relu',
                input_dim = X_num_columns))

model.add(Dense(90,
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(30,
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7,
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,
                activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
print("Model Created")

#Fit model to training data
model.fit(X_train, y_train, epochs=5000, batch_size=100)
print("Training completed")  

y_pred = model.predict(X_test)

rounded = [round(x[0]) for x in predictions]
print(rounded)



from sklearn import metrics
rmse = np.sqrt( metrics.mean_squared_error( y_test, y_pred ) )
round( rmse, 2 )
metrics.r2_score( y_test, y_pred )

model_json = model.to_json()
with open("model.json","w") as json_file:
 json_file.write(model_json)
model.save_weights("model.h5")




