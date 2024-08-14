#Project on Regression and Random Forest Regression
#Regression Problems in Machine Learning 
'''Machine Learning is a branch of Artificial Intelligence that enables computer programs to automatically learn and improve from experience.
Machine Learning Algorithms learn from datasets and then based on the patterns identified from the datasets make predictions on unseen data.
ML algorithms can be broadly categorized into two types:
1. Supervised Learning 
2. Unsupervised Learning
Supervised ML algorithms are those algorithms where the input dataset and the corresponding output or true prediction is available and the algorithms try to find the relationship between inputs and outputs.
In unsupervised ML algorithms, the true labels for the outputs are not known. Rather, the algorithms try to find similar patterns in the data. E.g., Clustering.
Supervised learning algorithms are further divided into two types:
1. Regression Algorithms
2. Classification Algorithms
Regression algorithms predict a continuous value e.g.,the price of a house.
Classification algorithms predict a discrete value e.g., whether a incoming email is Spam/Ham.'''

import pandas as pd 
import numpy as np 
import seaborn as sns
#sns.get_dataset_names()
#Importing the dataset and printing the dataset header
tips_df=sns.load_dataset("tips")
tips_df.head()
'''We will be using machine learning algorithms to predict the tip for a particular record based on the remaining features such as total_bill, gender, day, time etc.
Dividing Data into Features and Labels'''
x=tips_df.drop(['tip'],axis=1)
y=tips_df["tip"]
x.head()
y.head()

#Converting Categorical Data to Numbers
'''ML Algorithms can only work with numbers. It is important to convert categorical data into a numeric format'''
#Numeric Variables
numerical=x.drop(['sex','smoker','day','time'],axis=1)
numerical.head()
#DataFrame that contains only categorical columns
categorical=x.filter(['sex','smoker','day','time'])
categorical.head()
categorical["day"].value_counts()

'''One of the most common approaches to convert a categorical column to a numeric one is via one-hot encoding.
In one-hot encoding, for every unique value in the original columns, anew column is created.'''

cat_numerical=pd.get_dummies(categorical)
cat_numerical.head()
'''The final step is to join the numerical columns with the one-hot encoded columns.'''
x=pd.concat([numerical,cat_numerical],axis=1)
x.head()
#Divide Data into Training and Test Sets
'''We divide the dataset into two sets i.e., train and test set.
The dataset is trained via the train set and evaluated on the test set.'''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#Data Scaling/Normalization
'''The final step before data is passed to ML algorithm is to scale the data.
Some columns of the dataset contain small values, while the others contain very large values.It is better to convert all values to a uniform scale.'''

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
'''We have converted data into a format that can be sured to train ML algorithms for regression.'''