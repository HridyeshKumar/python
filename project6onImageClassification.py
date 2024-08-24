#Project on Image Classification/Recognition using CNN on CIFAR-10 Dataset
'''In this project we will be using CIFAR-10 dataset. This dataset includes thousands of pictures of 10 different kinds of objects like airplanes, automobiles, birds and so on.
Each image in the dataset includes a matching label so we know what kind of image it is.
The images in the CIFAR-10 dataset are only 32x32 pixels.'''
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
from tensorflow.keras.utils import to_categorical
#Load the dataset
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
#Normalize the data
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=255.0
X_test/=255.0
#Convert class vectors to binary class matrices
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
model=Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

#Compile the model 
model.compile(
   loss='categorical_crossentropy',
   optimizer='adam',
   metrics=['accuracy'])
model.summary()

#Train the model
model.fit(
   X_train,
   y_train,
   batch_size=32,
   epochs=25,
   validation_data=(X_test,y_test),
   shuffle=True)

#Save the neural network architecture
model_structure=model.to_json()
f=Path("model_structure.json")
f.write_text(model_structure)

#Save the trained neural network weights
model.save_weights("model_weight.h5")

#Making Predictions on the images
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
class_labels=["Planes","car","Bird","Cat","Deer","Dog","Frog","Horse","Boat","Truck"]
#load the json file that contains the model structure 
f=Path("model_structure.json")
model_structure=f.read_text()
#Recreate the keras model object from the json data
model=model_from_json(model_structure)
#Load an image file to test
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img,img_to_array
img=load_img("dog.png",target_size=(32,32))
plt.imshow(img)
#Convert the image to a numpy array 
from tensorflow.keras.utils import img_to_array
image_to_test=img_to_array(img)
list_of_images=np.expand_dims(image_to_test,axis=0)
#make predictions using the model 
results=model.predict(list_of_images)
#since we are only testing one image, we only need to check the first result 
single_result=results[0]
#We will get a likelihood score for all 10 possible classes.Find out which class has the highest score
most_likely_class_index=int(np.argmax(single_result))
class_likelihood=single_result[most_likely_class_index]
#Print the result 
print("This is a image of a {} likelihood:{:2f}".format(class_label,class_likelihood))