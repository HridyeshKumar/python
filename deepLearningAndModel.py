#Introduction to Deep Learning and Model on Iris Dataset
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
