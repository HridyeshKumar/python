+#Introduction to Deep Learning and Model on Iris Dataset
#Deep Learning Framework
'''Deep Learning is a field within Machine Learning that deals with building amd using Neural Network Models.
Neural Networks mimic the functioning of a human brain.
Neural Networks with more than three layers are typically categorised as Deep Learning Networks.'''
#Perceptron
'''The Perceptron is the unit of learning in an artificial neural networks.A Perceptron resembles a human brain cell.
A Perceptron is a single cell or node in a neural network.
In Deep Learning, we replace slope of model with weights called as w and intercept with the bias called as b.
Weights and Biases become the parameters for a neural network.
The number of weights equals the number of inputs/features.'''

#Artificial Neural Network
'''An ANN is a network of perceptrons. A deep neural network usually has three or more layers.
Each node has its own weights, biases and activation function.Each node is connected to all the nodes in the next layer forming a dense network.
Training an ANN means determining the right values for these parameters and hyperparameters such that it maximizes the accuracy of predictions for the given use case.'''

#Neural Network Architecture
#Input Layer
'''The input to Deep Learning model is usually a vector of Numeric values.
Vectors are usually defined using NumPy arrays. It represents the feature variables or independent variables that are used for prediction as well as training.'''
#Hidden Layer
'''An ANN can have one or more hidden layers. The more the layers are the deep the network is.
Each hidden layer can have one or more nodes. Typically, the node count is configured in range of 2^n. Ecamples are 8,16,32,64,128 etc.
A neural network is defined by the number of layers and nodes.
The output of each node in previous layer will become the input for every node in the current layers.
When there are more nodes and layers it usually results in better accuracy. As a general practice, start with small number and keep adding until an acceptable accuracy levels are obtained.'''
#Weights and Biases
'''They form the basis for Deep Learning Algorithms. Weights and Biases are trainable parameters in a neural network model.
Each input for each node will have an associated weight with it.'''
#Activation Functions
'''An activation function plays an important role in creating the output of the node in the neural network.
An activation function takes the matrix output of the node and determines if and how the node will propagate information to the next layer.
The main objective of activation function is that it converts the output to a non-linear value. They serve as a critical step in helping a neural network learn specific patterns in the data.
TanH:- A TanH function normalizes the output in the range of (-1 to +1)
ReLu:- Rectified Linear Unit- A ReLu produces a zero if the output is negative. Else, it will produce the same input verbatim.
Softmax Function:- This is used in the case of classification problems. It produces a vector of probabilities for each of the possible classes in the outcomes. The class with the highest probability will be considered as the final class.
These all activation functions are added as hyperparameters in the model.'''

#Output Layer
'''The output layer is the final layer in the neural network where desired predictions are obtained. '''
#Training a Neural Network Model
'''Set up and initialisation:- If error is high then it adjusts weights and biases by the process of gradient descent to improve accuracy.
Forward Propagation:- Movement from Input to hidden layer and then output layer.'''
#Measure Accuracy and Error 
'''Back Propagation:- If error is high then it adjusts weights amd biases by the process of gradient descent to improve accuracy.
Gradient Descent is the process of repeating the forward and backward propagation in order to reduce error and move closer to the desired model.
Batches and Epochs:- 10000/10(1000)
Validation and Testing'''

#Deep Learning Example- Iris Dataset
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

'''Prepare input data for deep learning
Load data into pandas dataframe
Convert the dataframe into numpy array
Scale the feature dataset
Use of one hot encoding for the target variable
Split the dataset into training and test datasets
Load Data and Review content'''

iris_data=pd.read_csv("iris.csv")
print(iris_data.head())
'''Use label encoder to convert String to Numeric values for the target variable'''
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
iris_data['Species']=label_encoder.fit_transform(iris_data['Species'])
print(iris_data.head())
#Converting input to numpy array
np_iris=iris_data.to_numpy()
print(np_iris.shape)
#Separate features and target variables
X_data=np_iris[:,0:4]
Y_data=np_iris[:,4]
print("\n Features before Scaling: \n---------")
print(X_data[:5,:])
print("\ntarget before one-hot ending: \n---------")
print(Y_data[:5])
#Create a standard scaler object that if fit on the input data
scaler=StandardScaler().fit(X_data)
#scale tge numeric feature variable
X_data=scaler.transform(X_data)
#convert target variable as a one-hot encoded array
Y_data=tf.keras.utils.to_categorical(Y_data,3)
print("\n Features after Scaling: \n----------")
print(X_data[:5,:])
print("\ntarget after one-hot encoding: \n----------")
print(Y_data[:5])
#Splitting the data into training and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.10)
print("\n Train test Dimensions: \n----------")
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
'''Create a Model
Number of hidden layers
Number of nodes in each layer 
Activation functions
Loss function and accuracy measurements. '''
from tensorflow import keras 
#Number of classes in the target variable
NB_CLASSES=3
#Create a sequential model in keras 
model=tf.keras.models.Sequential()
#add the first hidden layer 
model.add(keras.layers.Dense(128,#Number of nodes
input_shape=(4,),#number of input variables
name="Hidden-Layer-1",#Logical name
activation="relu"))#activation function
#add a second hidden layer
model.add(keras.layers.Dense(128,name="Hidden-Layer-2",activation="relu"))
#add an output layer with softmax function
model.add(keras.layers.Dense(NB_CLASSES,name="Output-Layer",activation="softmax"))
#compile the model with loss and metrics
model.compile(loss="categorical_crossentropy",metrics=["accuracy"])
#print the model summary
model.summary()
#Make it verbose so we can see the process 
VERBOSE=1
#Set hyperparameters for training 
#Set batch size
BATCH_SIZE=16
#Set the number of epochs
EPOCHS=20
#Set the validation split. 20% of the training dataset will be used for validation
VALIDATION_SPLIT=0.2
print("\nTraining Progress: \n------------")
'''Fitting the model. This will perform the entire training cycle, included forward propagation, loss computation, backward propagation and gradient descent.'''
history=model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
print("Accuracy During Training: \n------------")
import matplotlib.pyplot as plt
#Plot the accuracy of the model after each epoch
pd.DataFrame(history.history)["accuracy"].plot(figsize=(8,5))
plt.title("Accuracy improvement after each epoch")
plt.show()
#Evaluate the model against the test dataset and print the result
print("\nEvaluate against test dataset: \n------------")
model.evaluate(X_test,Y_test)
#Saving a model
model.save("iris_save")
#Load the model
loaded_model=keras.models.load_model("iris_save")
#print the model summary
loaded_model.summary()
#Predictions with Deep Learning Model
#raw prediction data
prediction_input=[[2.6,12.,2.4,4.4]]
#scale the prediction data with the same scaling object
scaled_input=scaler.transform(prediction_input)
#get the raw prediction probabilities
raw_prediction=loaded_model.predict(scaled_input)
print("Raw Prediction Output (Probabilities):",raw_prediction)
#Find Prediction
prediction=np.argmax(raw_prediction)
print("Prediction is",label_encoder.inverse_transform([prediction]))