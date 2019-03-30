from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os

sys.path.append(os.path.abspath('./model'))

from load import *
app = Flask(__name__)
global model_in
model_in = init()


@app.route('/')
def index():
 return render_template('index.html')
 
@app.route('/predict', methods=['GET'])
def predict():
 advt = pd.read_csv( "DemandDataSet_Pencil_Beans_Test.csv",header=None )
 X_test=advt
 print("My Model linreg",model_in)
 print("My Model linreg",model_rf)
 pred_y = model_in.predict( X_test )
 
 response=' '.join(map(str, pred_y))
 print("Predicted Demand is: Rs",pred_y[0])
 return response

def index():
 return render_template('index.html')
 
 
if __name__ =='__main__':
 app.run(debug=True,port=8085)

