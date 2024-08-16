#Project on Logistic Regression & Clustering
#Classification Problems in Machine Learning 
'''Classification problems are the type of problems where you have to predict a deiscrete value i.e., whether the student will pass the exam or not.'''

import pandas as pd 
import numpy as np
#importing the dataset
churn_df=pd.read_csv("Churn_Modelling.csv")
churn_df.head()

'''The exited column contains information regarding whether or not the customer exited the bank after six months.'''

#Removing unnecessary columns
churn_df=churn_df.drop(['RowNumber','CustomerId','Surname'],axis=1)
churn_df.head()
#Dividing Data into Features and Labels
X=churn_df.drop(['Exited'],axis=1)
y=churn_df['Exited']
X.head()
y.head()
#Converting Categorical Data to Numbers
numerical=X.drop(['Geography','Gender'],axis=1)
numerical.head()
categorical=X.filter(['Geography','Gender'])
categorical.head()
cat_numerical=pd.get_dummies(categorical)
cat_numerical.head()
X=pd.concat([numerical,cat_numerical],axis=1)
X.head()
#Dividing Data into Training and Test Sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Data Scaling/Normalization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

'''Binary Classification problems are those classification problems where there are only two possible values for the output level.
Whether a customer will leave the bank after a certain period or not.'''

#Logistic Regression 
'''Logistic Regression is a linear model, which makes classification by passing the output of linear regression through a sigmoid function.
Importing logistic regression classifier from sklearn'''