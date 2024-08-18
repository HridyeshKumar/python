#Project on Spam/Ham Classification using NLP
#Natural Language Processing- NLP
'''NLP is a field concerned with the ability of a computer to understand, analyze, manipulate and potentially generate human language.
NLP is a broad umbrella that encompasses many topics. Some of them are sentiment analysis, topic modelling, text classification etc.
NLTK:- Natural Language ToolKit: The NLTK is the most utilized package for handling natural language processing tasks. It is an open source library.'''
#Spam/Ham Classification using Natural Language Processing
#pip install NLTK
import nltk
import pandas as pd
import numpy as np
dataset=pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
dataset.columns=['label','body_txt']
dataset.head()
dataset['body_txt'][0]
dataset['body_txt'][1]
#What is the shape of the data
print("Input data has {} rows and {} columns".format(len(dataset),len(dataset.columns)))
#How many Spam/Ham are there
print("Out of {} rows,{} are spam and {} are ham".format(len(dataset),len(dataset[dataset['label']=='spam']),len(dataset[dataset['labal']=='ham'])))
#How much missing data is there
print("Number of null in label: {}".format(dataset['label'].isnull().sum()))
print("Number of null in text: {}".format(dataset['body_text'].isnull().sum()))
'''Preprocessing text data:- Cleaning up the text data is necessary to highlight attributes that you are going to use in ML algorithms.
Cleaning or preprocessing the data consists of a number of steps.
Remove Punctuation
Tokenization 
Remove Stopwords
Lemmatize/Stemming'''
import string 
string.punctuation
def remove_punct(text):
   text_nopunct="".join([char for char in text if char not in string.punctuation])
   return text_nopunct
dataset['body_txt_clean']=dataset['body_text'].apply(lambda x:remove_punct(x))
dataset.head()
#Tokenization
'''Tokenizing is splitting some string or sentence into a list of words'''
import re
def tokenize(text):
   tokens=re.split('\W',text)
   return tokens
dataset['body_text_tokenized']=dataset['body_text_clean'].apply(lambda x:tokenize(x.lower()))
dataset.head()
'''Remove Stopwords:- These are commonly used words like the, and, but,if that don't contribute much to the meaning of a sentence.'''
stopwords=nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
   text=[word for word in tokenized_list if word not in stopwords]
   return text
dataset['body_text_nostop']=dataset['body_text_tokenized'].apply(lambda x:remove_stopwords(x))
dataset.head()

