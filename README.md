# salesPrediction
Sales Prediction using Flask and Python

For running the app in localhost
Anaconda root->Open Terminal
Get inside the veg folder
python app.py
open localhost:8085
Flask Browser prediction for Linear Regression.
	For all three(Linear Regression/Random Forest/DNN using Keras/TensorFlow) run respective .py files(CreateModel*.py) 
  You may use spider or any python console(editor)
  	 Create the models
     Run TestModel.py
     Yay!! gives you all three predictions.
Modify the code as demand.
Please Cite the github link https://github.com/vmrdigital/salesPrediction/ in your work.


Structure
5 layers:

Input layer: 300 relu neurons with no dropout
1st hidden layer: 90 relu neurons with 20% dropout
2nd hidden layer: 30 relu neurons with 20% dropout
3rd hidden layer: 7 relu neurons with 20% dropout
Output layer: 1 linear relu
optimizer used: adam loss measured using mean squared error

Training
5000 epochs using batch size 100
Model Accuracy
Accuracy : 0.92

Train
Accuracy (Train): 0.95

Test
Accuracy (Test): 0.92

Production
Accuracy (Production): 0.90
    
