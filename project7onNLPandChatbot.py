#Project on Sentiment Analysis using NLP and Chatbot using NLP
#Sentiment Classification using NLP and Classification Algorithm
'''Sentiment Analysis is a means to identify the view or emotion behind a situation.
It basically means to analyze and find the emotion or intent behind a piece of text or speech or any model of communication.
This burger has a very bad taste- negative review
I ordered this pizza today- neutral sentiment/review
I love this cheese sandwich, its so delicious- positive review'''
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_score,roc_curve
from sklearn.metrics import classification_report, plot_confusion_matrix

df_train=pd.read_csv("train.txt",delimiter=";",names=['text','label'])
df_val=pd.read_csv("val.txt",delimiter=";",names=['text','label'])

df=pd.concat([df_train,df_val])
df.reset_index(inplace=True,drop=True)
print("Shape of the dataframe:",df.shape)
df.sample(5)

import warnings
warnings.filterwarnings("ignore")
sns.countplot(df.label)

'''Positive Sentiment- joy,love,surprise
Negative Sentiment- anger,sadness,fear
Now we will create a custom encoder to convert categorical target labels to numerical i.e. 0 and 1'''

def custom_encoder(df):
   df.replace(to_replace="surprise",value=1,inplace=True)
   df.replace(to_replace="love",value=1,inplace=True)
   df.replace(to_replace="joy",value=1,inplace=True)
   df.replace(to_replace="fear",value=0,inplace=True)
   df.replace(to_replace="anger",value=0,inplace=True)
   df.replace(to_replace="sadness",value=0,inplace=True)
custom_encoder(df['label'])
sns.countplot(df.label)
'''Preprocessing Steps:-
Get rid of any characters apart from alphabets
Convert the string to lowercase because Python is case-sensitive
3 check and remove the stopwords
Perform Lemmatization
