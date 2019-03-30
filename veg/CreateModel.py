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

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit( X_train, y_train )
linreg.intercept_
list( zip( ["v","p","r","h","pr","se","st","ev","we","ve","lo","su","qt","pr1","su1","wed","fe","na","cl","ex"], list( linreg.coef_ ) ) )
y_pred = linreg.predict( X_test )
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
joblib.dump(linreg, 'lin_model.pkl', compress=9)



