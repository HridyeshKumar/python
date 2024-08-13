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
titanic_data.isnull().sum()
median=titanic_data.age.median()
print(median)
mean=titanic_data.age.mean()
print(mean)
titanic_data["Median_Age"]=titanic_data.age.fillna(median)
titanic_data["Mean_Age"]=titanic_data.age.fillna(mean)
titanic_data["Mean_Age"]=np.round(titanic_data["Mean_Age"],1)
titanic_data.head(20)

#Frequent Category Imputation
'''One of the most common ways of handling missing values in a categorial column is to replace the missing values with the most frequent occuring values i.e., the mode of the column.'''

import matplotlib.pyplot as plt
import seaborn as sns

titanic_data=sns.load_dataset("titanic")
titanic_data=titanic_data[["embark_town","age","fare"]]
titanic_data.head()
titanic_data.isnull().mean()
titanic_data.embark_town.value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel("Embark Town")
plt.ylabel("Number of Passengers")
titanic_data.embark_town.mode()
titanic_data.embark_town.fillna("Southampton",inplace=True)

#Categorial Data Encoding
'''Models based on statistical algorithms such as machine learning and deep learning work with numbers.
A dataset can contain numerical, categorical, datetime, and mixed variables.
A mechanism is needed to convert categorical data to its numeric counterpart so that the data can be used to build statistical models.
The techniques used to convert numeric data into categorical data are called categorical data encoding schemes.'''

#One Hot Encoding
'''One Hot Encoding is one of the most commonly used categorical encoding schemes.
In one hot encoding for each unique value in the categorical column a new column is added.
Integer 1 is added to the column that corresponds to the original label and all the remaining column are filled with zeros.'''

titanic_data=sns.load_dataset("titanic")
titanic_data.head()
titanic_data=titanic_data[["sex","class","embark_town"]]
titanic_data.head()

import pandas as pd 
temp=pd.get_dummies(titanic_data["sex"])
temp.head()
pd.concat([titanic_data["sex"],pd.get_dummies(titanic_data["sex"])],axis=1).head()
temp=pd.get_dummies(titanic_data["embark_town"])
temp.head()

#Label Encoding
'''In label encoding, labels are replaced by integers.
That is why label encoding is also called as Integer Encoding.'''

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(titanic_data["class"])
titanic_data["le_class"]=le.transform(transform(titanic_data["class"])
titanic_data.head()

#Data Discretization
'''The process of converting continuous numeric values such as price, age, and weight into discrete intervals is called discretization or binning.'''

#Equal Width Discretization
'''The most common type of discretization approach is fixed width discretization.'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
diamond_data=sns.load_dataset("diamonds")
diamonds_data.head()
sns.distplot(diamond_data["price"])

'''The histogram for price column shows that the data is positively skewed.'''

price_range=diamond_data["price"].max()-diamond_data["price"].min()
print(price_range)
price_range/10
lower_interval=int(np.floor(diamond_data["price"].min()))
upper_interval=int(np.ceil(diamond_data["price"].max()))
interval_length=int(np.round(price_range/10))
print(lower_interval)
print(upper_interval)
print(interval_length)
total_bins=[i for i in range(lower_interval,upper_interval+interval_length,interval_length)]
print(total_bins)
bin_labels=['Bin_no_'+str(i) for i in range(l,len(total_bins))]
print(bin_labels)
diamond_data['price_bins']=pd.cut(x=diamond_data['price'],bins=total_bins,labels=bin_labels,include_lowest=True)
diamond_data.head()
diamond_data.groupby('price_bins')['price'].count().plot.bar()
plt.xticks(rotation=45)
'''The output shows that the price of most of the diamonds lies in the first bin or the first interval.'''

#Handling Outliers
'''Outliers  are the values that are too far from the rest of the observation.'''

#Outlier Trimming
'''As the name suggests it refers to simply removing the outliers beyond a certain threshold value.
One of the main advantage of outlier trimming is that it is extremely quick and doesn't distort the data.'''

titanic_data=sns.load_dataset("titanic")
titanic_data.head()
sns.boxplot(y="age",data=titanic_data)

'''One of the most common ways to remove the outliers is to find the Inter Quartile Range(IQR), multiply it  by 1.5 and then subtract it from the first quartile value(0.25 quantile).
To find the upper limit, add the product of IQR and 1.5 to the 3rd quartile value(0.75 quantile)'''
 