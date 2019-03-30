# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:47:46 2019

@author: Ranajoy
"""
import pandas as pd
import numpy as np

advt = pd.read_csv( "DemandDataSet_Pencil_Beans.csv" )

advt = advt[["v","p","r","h","pr","se","st","ev","we","ve","lo","su","qt","pr1","su1","wed","fe","na","cl","ex","Quantity"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
  advt[["v","p","r","h","pr","se","st","ev","we","ve","lo","su","qt","pr1","su1","wed","fe","na","cl","ex"]],
  advt.Quantity,
  test_size=0.3,
  random_state = 42 )
  
  

from sklearn.ensemble.forest import RandomForestRegressor
# build our RF model
RF_Model = RandomForestRegressor(n_estimators=100,
                                 max_features=1, oob_score=True)
# let's get the labels and features in order to run our 
# model fitting
labels = y_train
features = X_train
rgr=RF_Model.fit(features, labels)



X_test_predict=pd.DataFrame(
    rgr.predict(X_test)).rename(
    columns={0:'predicted_price'}).set_index('predicted_price')
X_train_predict=pd.DataFrame(
    rgr.predict(X_train)).rename(
    columns={0:'predicted_price'}).set_index('predicted_price')
# combine the training and testing dataframes to visualize
# and compare.
RF_predict = X_train_predict.append(X_test_predict)

y_pred = rgr.predict( X_test )
test_pred_df = pd.DataFrame( { 'actual': y_test,
                            'predicted': np.round( y_pred, 2),
                            'residuals': y_test - y_pred } )
test_pred_df[0:10]



from sklearn import metrics
rmse = np.sqrt( metrics.mean_squared_error( y_test, y_pred ) )
round( rmse, 2 )
metrics.r2_score( y_test, y_pred )
import matplotlib.pyplot as plt
import seaborn as sn
residuals = y_test - y_pred
sn.jointplot(  advt.Quantity, residuals, size = 6 )
sn.plt.show()
sn.distplot( residuals )
sn.plt.show();
import pickle
from sklearn.externals import joblib
joblib.dump(rgr, 'rf_model.pkl', compress=9)


