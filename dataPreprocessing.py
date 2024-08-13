#Missing Value Treatment, Data Discretization, Feature Selection using Variance & Correlation

#Data Preprocessing
'''Data Preprocessing involves cleaning and engineering data in a way that it can be used as input to several important data science tasks such as data visualization, machine learning, deep learning, and data analytics.
Some of the most common data preparation tasks include feature scaling, handling missing values, categorial variable encoding, data discretization.'''

#Feature Scaling 
'''A dataset can have different attributes. The attributes can have different magnitudes, variances, standard deviation, mean value etc.
For instance, salary can be in thousands, whereas age is normallly a two-digit number.
The difference in the scale or magnitude of attributes can actually affect statistical models.
For instance, variables wirh bigger ranges dominate those with smaller ranges for linear models.'''

#Standardization
'''Standardization is the process of centering a variable at zero and standardizing the data variance to 1.
To standardize a dataset, you simply have to subtract each data point from the mean 
of all the data points and divide the d
result by the standard deviation of the data.
Feature Scaling is applied on numeric data only.'''

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
titanic_data=sns.load_dataset("titanic")
titanic_data=titanic_data["age","fare","prices"]]
titanic_data.head()
titanic_data.describe()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(titanic_data)
titanic_data_scaled=scaler.transform(titanic_data)
titanic_data_scaled=pd.DataFrame(titanic_data_scaled,columns=titanic_data.columns)
titanic_data_scaled.head()
sns.kdeplot(titanic_data_scaled["age"])

#Min/Max Scaling
'''In min/max scaling, you subtract each value by the minimum value and divide the result by the difference between minimum and maximum value in the dataset.'''

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(titanic_data)
titanic_data_scaled=scaler.transform(titanic_data)
titanic_data_scaled=pd.DataFrame(titanic_data_scaled,columns=titanic_data.columns)
titanic_data_scaled.head()
sns.kdeplot(titanic_data_scaled["age"])

#Handling Missing Data
'''Missing values are those observations in the dataset that do not contain any value.
Missing values can totally change data patterns and therefore it is extremely important to understand why missing values occur in the dataset and how to handle them.'''

#Handling Missing Numerical Data
'''To handle missing numerical data, we can usee statistical techniques. The use of statistical techniques or algorithms to replace missing values with statistically generated values is called imputation'''

titanic_data=sns.load_dataset("titanic")
titanic_data.head()
titanic_data=titanic_data[["survived","pclass","age","fare"]]
titanic_data.head()
titanic_data.isnull().mean()
