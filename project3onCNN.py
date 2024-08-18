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
'''Converting the target value into 10 bins. So, we will see that the output from a model will then go into one of these bins.'''
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
print(y_train.shape)
print(y_test.shape)
print(y_train[0])
#Building the model
model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation="softmax"))
#Compile the model
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=["accuracy"])
model.summary()
history=model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))
plt.plot(history.history['accuracy'])
#Evaluating the model
score=model.evaluate(X_test,y_test)
'''In neural networks, we only have fully connected layer, otherwise known as dense layer. With Convolutional Neural Networks, we have more operations such as the convolution operation, max pooling, flattening and also a fully connected layer.'''
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255.0
X_test/=255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#CNN Model Development
cnn=Sequential()
cnn.add(Conv2D(32,kernal_size=(3,3),input_size=(28,28,1),padding='same',activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(32,kernal_size=(3,3),padding='same',activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Flatten())
cnn.add(Dense(64,activation='relu'))
cnn.add(Dense(10,activation='softmax'))
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(cnn.summary())
history_cnn=cnn.fit(X_train,y_train,epochs=12,verbose=1,validation_data=(X_train,y_train))
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])