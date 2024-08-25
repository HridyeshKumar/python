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
Perform Lemmatization'''

lm=WordNetLemmatizer()
def text_transformation(df_col):
   corpus=[]
   for item in df_col:
      new_item=re.sub('[^a-zA-Z]',' ',str(item))
      new_item=new_item.lower()
      new_item=new_item.split()
      new_item=[lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
      corpus.append(' '.join(str(x) for x in new_item))
   return corpus
corpus=text_transformation(df['text'])
cv=CountVectorizer(ngram_range=(1,2))
traindata=cv.fit_transform(corpus)
x=traindata
y=df.label
'''Now we will fit the data into grid search and view the best parameters using the best_params attribute'''
parameters={'max_features':('auto','sqrt'),'n_estimators':[5,10],'max_depth':[10,None],'min_samples_leaf':[5],'min_samples_leaf':[1],'bootstrap':[True]}
grid_search=GridSearchCV(RandomForestClassifier(),parameters,cv=5,return_train_score=True,n_jobs=-1)
grid_search.fit(x,y)
grid_search.best_params_
'''We can view all the models and their respective parameters,mean test score and rank as GridSearch CV'''
for i in range(8):
   print('Parameters:',grid_search.cv_results_['params'][i])
   print('Mean test Score:',grid_search.cv_results_[mean_test_score'][i])
   print("Rank:",grid_search.cv_results_['rank_test_score'])

'''Now we will choose the best parameter obtained from GridSearchCV and create a final random forest classifier model and then train our model.'''

rfc=RandomForestClassifier(max_features=grid_search.best_params_['max_features'],max_depth=grid_search.best_params_['max_depth'],n_estimators=grid_search.best_params_['n_estimators'],min_samples_split=grid_search.best_params_['min_samples_split'],min_samples_leaf=grid_search.best_params_['min_samples_leaf'],bootstrap=grid_search.best_params_['bootstrap'])
rfc.fit(x,y)

#Test Data Transformation
test_df=pd.read_csv('test.txt',delimiter=';',names=['text','label'])
X_test,y_test=test_df.text,test_df.label
#encode the labels into two classes 0 and 1
test_df=custom_encoder(y_test)
#preprocessing of text
test_corpus=text_transformation(X_test)
#convert the text data into vectors
testdata=cv.transform(test_corpus)
#predict the target
predictions=rfc.predict(testdata)

#Model Evaluation
'''We will evaluate our model using various metrics such as accuracy score, recall score confusion matrix.'''

acc_score=accuracy_score(y_test,predictions)
pre_score=precision_score(y_test,predictions)
rec_score=recall_score(y_test,predictions)
print('Accuracy Score:',acc_score)
print('Precision Score:',pre_score)
print('Recall Score:',rec_score)
print("-"*50)
cr=classification_report(y_test,predictions)
print(cr)
'''ROC Curve- We will plot probability of the class using the predict_proba() method of random forest classifier
and then we will plot the curve.'''
predictions_probability=rfc.predict_proba(testdata)
fpr,tpr,thresfolds=roc_curve(y_test,predictions_probability[:,1])
plt.plot(fpr,tpr)
plt.plot([0,1])
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
'''As we can see that our model performed very well in classifying the sentiments, with an accuracy score, precision score and recall score of approx 96%
Now we will check for custom input as well and let our model identity the sentiment of the input statement.'''

def expression_check(prediction_input):
   if prediction_input==0:
      print("Input statement has negative sentiment")
   elif prediction_input==1:
      print("Input statement has positive sentiment")
   else:
      print("Invalid Statement")
'''Function to take the input statement and performs the same transformation as we did earlier'''
def sentiment_predictor(input):
   input=text_transformation(input)
   transformed_input=cv.transform(input)
   predictions=rfc.predict(transformed_input)
   expression_check(prediction)
input1=["Sometimes I just don't want to go out"]
input2=["I bought a new phone and it's so good"]
sentiment_predictor(input1)
sentiment_predictor(input2)
'''Input statement has negative sentiment
Input statement has positive statement'''

#Chatbot using NLP and Neural Networks in Python
'''Tag means classes
Patterns means what user is going to ask
Response is chatbot reponse'''
data={"intents":[{"tag":"greetings","patterns":["Hello","How are you?","Hi There","Hi", "What's up"],"responses":["Howdy Partner!","Hello","How are you doing?","Greetings!","How do you do"]},{"tag":"age","patterns":["how old are you","when is your birthday","when was you born"],"responses":["I am 24 years old","I was born in 1966","My birthday is July 3rd and I was born in 1996","03/07/1996"]},{"tag":"date","patterns":["what are you doing this weekend","do you want to hangout sometime?","what are your plans for this week"],"responses":["I am available this week","I don't have any plans","I am not busy"]},{"tag":"name","patterns":["what's your name","what are you called","who are you"],"responses":["My name is Kippi","I'm Kippi","Kippi"]},{"tag":"goodbye","patterns":["bye","g2g","see ya","adios","cya"],"responses":["It was nice speaking to you","See you later","Speak Soon"]},]}
'''For each tag we created, we would specify patterns. Essentially this defines the different ways of how a user may pose a query to the chatbot.
The chatbot would then take these patterns and use them as training data to determine what someone is asking and the chatbot reponse would be relevant to that question.'''
import json
import string
import random
import nltk
import numpy as np 
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout
nltk.download("punkt")
nltk.download("wordnet")
'''In order to create our training data below steps to be followed
Create a vocabulary of all the words used in the patterns
Create a list of the classes-tag of each intent
Create a list of all the patterns within the intents file
Create a list of all the associated tags to go with each patterns in the intents file.
Initializing lemmatizer to get stem of words'''
lemmatizer=OrdNetLemmatizer()
words=[]
classes=[]
doc_x=[]
doc_y=[]
'''Loop through all the intents 
Tokenize each pattern and append token to words, the patterns and the associated tag to their associated list'''
for intent in data["intents"]:
   for pattern in intent["patterns"]:
      tokens=nltk.word_tokenize(pattern)
      words.extend(tokens)
      doc_x.append(pattern)
      doc_y.append(intent["tag"])
   if intent["tag"] not in classes:
      classes.append(intent["tag"])
#Lemmatize all the words in the vocab and convert them to lowercase
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
'''Sorting the vocab and classes in alphabetical order and taking the set to ensure no duplicates occur'''
words=sorted(set(words)
classes=sorted(set(classes))
print(words)
print(classes)
print(doc_x)
print(doc_y)
#List for training data
training=[]
out_empty=[0]*len(classes)
#creating a bag of words model
for idx,doc in enumerate(doc_x):
   bow=[]
   text=lemmmatizer.lemmatize(doc.lower())
   for word in words:
      bow.append(1) if word in text else bow.append(0)
   output_row=list(out_empty)
   output_row[classes.index(doc_y[idx])]=1
   training.append([bow,output_row])
random.shuffle(training)
training=np.array(training,dtype=object)
train_X=np.array(list(training[:,0]))
train_y=np.array(list(training[:,1]))
'''The model will look at the features and predict the tag associated with the features and then will select an appropriate message/response from the tag.'''
input_shape=(len(train_X[0]),)
output_shape=len(train_y[0])
epochs=500
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
#Create a Sequential model
model=Sequential()
model.add(Dense(128,input_shape=input_shape,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_shape,activation='softmax'))
#Create the Adam optimizer with a specified learning rate
adam=tf.keras.optimizers.Adam(learning_rate=0.01)
#compile the model using the Adam optimizer
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
print(model.summary())
model.fit(x=train_X,y=train_y,epochs=500,verbose=1)
def clean_text(text):
   tokens=nltk.word_tokenize(text)
   tokens=[lemmatizer.lemmatize(word) for word in tokens]
   return tokens
def bag_of_words(text,vocab):
   tokens=clean_text(text)
   bow=[0]*len(vocab)
   for w in tokens:
      for idx,word in enumerate(vocab):
         if word==w:
            bow[idx]=1
   return np.array(bow)
def pred_class(text,vocab,labels):
   bow=bag_of_words(text,vocab)
   result=model.predict(np.array([bow]))[0]
   thresh=0.2
   y_pred=[[idx,res] for idx,res in enumerate(result) if res>thresh]
   y_pred.sort(key=lambda x:x[1],reverse=True)
   return_list=[]
   for r in y_pred:
      return_list.append(labels[r[0]])
   return return_list
def get_response(intents_list,intent_json):
   tag=intents_list[0]
   list_of_intents=intents_json["intents"]
   for i in list_of_intents:
      if i["tag"]==tag:  
         result=random.choice(i["responses"])
         break
   return result
#Running the chatbot
while True:
   message=input("")
   intents=pred_class(message,words,classes)
   result=get_response(intents,data)
   print(result)