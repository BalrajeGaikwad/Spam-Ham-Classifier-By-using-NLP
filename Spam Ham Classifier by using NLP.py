# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 17:41:22 2025

@author: Admin
"""

import pandas as pd
import numpy as np

message = pd.read_csv(
    r'C:\Users\Admin\Downloads\sms+spam+collection\SMSSpamCollection',
    sep='\t',
    names=['label', 'message']
)

import nltk
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

ps=PorterStemmer()
corpus=[]

for i in range(0, len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=" ".join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(message['label'],dtype=int)
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y , test_size=0.25, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_dataset_model=MultinomialNB().fit(x_train,y_train)
y_pred=spam_dataset_model.predict(x_test)

y_pred

from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test, y_pred)
