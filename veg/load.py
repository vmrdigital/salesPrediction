from sklearn.externals import joblib
import pandas as pd


def init():
	print("into init:")
	advt = pd.read_csv( "DemandDataSet_Pencil_Beans_Test.csv",header=None )
	X_test=advt
	print("Loaded Model from disk")
	model_in = joblib.load('lin_model.pkl')
	
	print(model_in)
	return model_in