#Project on Image Classification using Convolutional Neural Networks
#Convolutional Neural Networks
'''A Convolutional Neural Network (CNN) is a type of artificial neural network that is used in image recognition and processing that is specifically designed to process pixel data.'''
#CNN Model on MNIST Dataset for written digit classification
'''MNIST Dataset is the handwritten numbers taken as images. All images are grey scale.'''
from keras.datasets import mnist
#from keras.preprocessing.image import load_img, array_to_img
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
#Load the data
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#Understand the image format
X_train[0].shape
plt.imshow(X_train[0],cmap="gray")
y_train[0]
#Preprocessing the image data
image_height,image_width=28,28
X_train=X_train.reshape(60000,image_height*image_width)
X_test=X_test.reshape(10000,image_height*image_width)
print(X_train.shape)
print(X_test.shape)
print(X_train[0])
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255.0
X_test/=255.0
print(X_train[0])
print(y_train.shape)
print(y_test.shape)