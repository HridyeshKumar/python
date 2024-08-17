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
 
from sklearn.linear_model import LogisticRegression
log_clf=LogisticRegression()
classifier=log_clf.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

'''There are various metrics to evaluate a classification method.
Some of the most commonly used classification metrics are F1 score, recall, precision, accuracy and confusion matrix.
True Negatives(TN/tn):True Negatives are those output labels that are actually false and the model also predicted them as false.
True Positives(TP/tp):True Positives are those output labels that are actually true and the model also predicted them as true.
False Negatives(FN/fn):False Negatives are those output labels that are actually true but the model predicted them as false.
False Positives(FP/fp):False Positives are those output labels that are actually false but the model also predicted them as true.'''

#Precision
'''It is obtained by dividing true positives by the sum of true positive and false positive.
Precision=tp/(tp+fp)'''
#Recall
'''It is obtained by dividing true positives by the sum of true positives and false negatives.
Recall=tp/(tp+fn)'''

#Evaluating the algorithm on the test set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier(random_state=42,n_estimators=500)
classifiers=rf_clf.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#Clustering
'''Clustering algorithms are unsupervised algorithms where the training data is not labeled.
Rather, the algorithms cluster or group the datasets based on common characteristics.'''

#K-Means Clustering
'''K-Means Clustering is one of the most commonly used algorithms for clustering, K refers to the number of clusters that you want your data to be grouped into.
In K-Means clustering, the number of clusters has to be defined before K clustering can be applied to the data points.'''

#Steps for K-Means Clustering
'''1.Randomly assign centroid values for each cluster.
2.Calculate the euclidean distance between each data point and centroid values of all the clusters.
3.Assign the data point to the cluster of the centroid with the shortest distance.
4.Calculate and update centroid values based on the mean values of the coordinates of all the data points of the corresponding cluster.
5.Repeat steps 2-4 until new centroid values for all the clusters are different from the previous centroid values.'''

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Customer Segmentation using K-Means Clustering
'''In this project, you will see how to segment customers based on their incomes and past spending habits.
You will then identify customers who have high incomes and higher spending.'''

dataset=pd.read_csv("Mall_Customers.csv")
dataset.head()
'''The output shows that the dataset contains 200 records and 5 cloumns.
Plotting the histogram for the annual income column.'''
import warnings
warnings.filterwarnings("ignore")
sns.distplot(dataset["Annual Income (k$)"],kde=False,bins=50)
'''The output shows that most of the customers have incomes between 60 and 90K per year.
Plotting the histogram for the spending score column.'''
sns.distplot(dataset["Spending Score (1-100)"],kde=False,bins=50,color="red")
'''The output shows that most of the customers have a spending score between 40 and 60.
Plotting regression plot for annual income against spending score.'''
sns.regplot(x="Annual Income (k$)",y="Spending Score (1-100)",data=dataset)
'''There is no linear relationship between annual income and spending.
Plotting regression plot for age and spending score'''
sns.regplot(x="Age",y="Spending Score (1-100)",data=dataset)
'''The output confirms an inverse linear relationship between age and spending score.
Young people have higher spending compared to older people.'''
dataset=dataset.filter(["Annual Income(k$)","Spending Score (1-100)"],axis=1)
dataset.head()
km_model=KMeans(n_clusters=4)
km_model.fit(dataset)
print(km_model.cluster_centers_)
print(km_model.labels_)
plt.scatter(dataset.values[:,0],dataset.values[:,1],c=km_model.labels_,cmap='rainbow')
plt.scatter(km_model.cluster_centers_[:,0],km_model.cluster_centers_[:,1],s=100,c='black')
#Elbow method to get the optimal number of cluaters
loss=[]
for i in range(1,11):
   km=KMeans(n_clusters=i).fit(dataset)
   loss.append(km.inertia_)
plt.plot(range(1,11),loss)
plt.title('Finding optimal number of vlusters via elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('loss')
plt.show() 
km_model=KMeans(n_clusters=5)
km_model.fit(dataset)
print(km_model.cluster_centers_)
print(km_model.labels_)
plt.scatter(dataset.values[:,0],dataset.values[:,1],c=km_model.labels_,cmap='rainbow')
plt.scatter(km_model.cluster_centers_[:,0],km_model.cluster_centers_[:,1],s=100,c='black')
#Filtering all records with cluster id 1
cluster_map=pd.DataFrame()
cluster_map['data_indx']=dataset.index.values
cluster_map['cluster']=km_model.labels_
print(cluster_map)
cluster_map=cluster_map[cluster_map.clusters==1]
cluster_map.head()
'''These are the customers who have high incomes and high spending and these customers should be targeted during marketing campaigns.'''
