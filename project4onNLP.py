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
'''Stemming:- Stemming is the process of reducing inflected or derived words to their stem or root.'''
ps=nltk.PorterStemmer()
def stemming(tokenized_text):
   text=[ps.stem(word) for word in tokenized_text]
   return text
dataset['body_text_stemmed']=dataset['body_text_nostop'].apply(lambda x:stemming(x))
dataset.head()
'''Lemmatization:- It is the process of grouping together the inflected forms of a word so they can be analysed as a single term, identified by the word's lemma.For e.g. type, typing and typed are forms of the same lemma type.'''
wn=nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
   text=[wn.lemmatize(word) for word in tokenized_text]
   return text
dataset['body_text_lemmatized']=dataset['body_text_nostop'].apply(lambda x:lemmatizing(x))
dataset.head()
'''Vectorization:- This is defined as the process of encoding text as integers to create feature vectors. In out ontext we will be taking individual text messages and converting it to a numeric vector that represents that text message.
Count Vectorization:- This creates a document-term matrix where the entry of each cell will be a count of the number of times that word occured in that document.'''
from sklearn.feature_extraction.text import CountVectorizer
def clean_text(text):
   text="".join([word.lower() for word in text if word not in string.punctuation])
   tokens=re.split('\W',text)
   text=[ps.stem(word) for word in tokens if word not in stopwords]
   return text
count_vect=CountVectorizer(analyzer=clean_text)
X_count=count_vect.fit_transform(dataset['body_text'])
print(X_count.shape)
#Apply count vectorizer to a smaller sample
data_sample=dataset[0:20]
count_vect_sample=CountVectorizer(analyzer=clean_text)
X_count_sample=count_vect_sample.fit_transform(data_sample['body_text'])
print(X_count_sample.shape)
'''Sparse Matrix:- A matrix in which most entries are 0. In the interest of efficient storage, a sparse matrix will be stored by only storing the locations of the non-zero elements.'''
print(X_count_sample)
X_counts_df=pd.DataFrame(X_count_sample.toarray())
print(X_counts_df)
'''TF-IDF(Term Frequency,Inverse Document Frequency):- Creates a document term matrix where the column represents single unique terms(unirams) but the cell represents a weighting meant to represent how important a word is to a document.'''
from sklearn .feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer(analyzer=clean_text)
X_tfidf=tfidf_vect.fit_transform(dataset['body_text'])
print(X_tfidf.shape)
#Apply TfidfVectorizer to a smaller sample
data_sample=dataset[0:20]
tfidf_vect_sample=TfidfVectorizer(analyzer=clean_text)
X_tfidf_sample=tfidf_vect_sample.fit_transform(data_sample['body_text'])
print(X_tfidf_sample.shape)
X_tfidf_df=pd.DataFrame(X_tfidf_sample.toarray())
X_tfidf_df.columns=tfidf_vect_sample.get_feature_names()
print(X_tfidf_df)

#Feature Engineering: Feature Creation
dataset=pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
dataset.columns=['label','body_text']
dataset.head()
#Create feature for text message length
dataset['body_len']=dataset['body_text'].apply(lambda x:len(x)-x.count(" "))
dataset.head()
#create feature for % of text that is punctuation
def count_punct(text):
   count=sum([1 for char in text if char in string.punctuation])
   return round(count/(len(text)-text.count(" ")),3)*100
dataset['punct%']=dataset['body_text'].apply(lambda x:count_punct(x))
dataset.head()
import matplotlib.pyplot as plt 
import numpy as np
bins=np.linspace(0,200,40)
plt.hist(dataset['body_len'],bins)
plt.title('Body Length Distribution')
plt.show()
bins=np.linspace(0,50,40)
plt.hist(dataset['punct%'],bins)
plt.title('Punctuation % Distribution')
plt.show()

#Building Machine Learning Classifiers using Random Forest Model
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
dataset=pd.read_csv("SMSSpamCollection.tsv",sep="\t",header=None)
dataset.columns=['label','body_text']
dataset.head()
def count_punct(text):
   count=sum([1 for char in text if char in string.punctuation])
   return round(count/(len(text)-text.count(" ")),3)*100
dataset['punct%']=dataset['body_text'].apply(lambda x:count_punct(x))
dataset['body_len']=dataset['body_text'].apply(lambda x:len(x)-x.count(" "))
dataset.head()
def clean_text(text):
   text="".join([word.lower() for word in text if word not in string.punctuation])
   tokens=re.split('\W',text)
   text=[ps.stem(word) for word in tokens if word not in stopwords]
   return text
tfidf_vect=TfidfVectorizer(analyzer=clean_text)
X_tfidf=tfidf_vect.fit_transform(dataset['body_text'])
X_feaures=pd.concat([datset['body_len'],dataset['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_feaures.head()

#Model using K-Fold cross validation
from sklearn.ensemble import RandomForestClassifer
from sklearn.model_selection import KFold, cross_val_score
rf=RandomForestClassifier(n_jobs=1)
k_fold=KFold(n_splits=5)
cross_val_score(rf,X_features,dataset['label'],cv=k_fold,scoring='accuracy',n_jobs=1)

#Model using Train Test Split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_features, dataset['label'],test_size=0.3,random_state=0)
rf=RandomForestClassifier(n_estimators=500,max_depth=20,n_jobs=-1)
rf_model=rf.fit(X_train,y_train)
sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[0:10]
y_pred=rf_model.predict(X_test)
precision,recall,fscore,support=score(y_test,y_pred,pos_label='spam',average='binary')
print('Precision {} / Recall {} /Acccuracy {}'.format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_pred),3)))
