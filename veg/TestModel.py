# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:47:46 2019

@author: Ranajoy
"""
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf

advt = pd.read_csv( "DemandDataSet_Pencil_Beans_Test.csv",header=None )
X_test=advt


model_clone = joblib.load('lin_model.pkl')
pred_y = model_clone.predict( X_test )

model_clone_rf = joblib.load('rf_model.pkl')
pred_z = model_clone_rf.predict( X_test )




print("Predicted Demand using Multivariate Linear Regression is: ",pred_y[0])
print("Predicted Demand using Multivariate Random Forest is: ", pred_z[0])

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

loaded_model.compile(loss='mean_squared_error', optimizer='adam')


pred_a = loaded_model.predict( X_test )
print("Predicted Demand using DNN(Keras/TensorFlow) is: ", pred_a[0])

