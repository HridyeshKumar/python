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
