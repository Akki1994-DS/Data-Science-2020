# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 02:24:45 2020

@author: axays
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import re

Mail=pd.read_csv('C:\\Users\\axays\\Downloads\\ham_spam.csv',encoding='latin1')


def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

"I am tangocharlie 007 @#$%%%^ copy and out".split(" ")

# testing above function with sample text => removes punctuations, numbers
cleaning_text("Everyting comes with a price think twice before you do something wrong.")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .. are you?")

Mail.text = Mail.text.apply(cleaning_text)

# removing empty rows 
Mail.shape
Mail = Mail.loc[Mail.text != " ",:]

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
x_train,y_test = train_test_split(Mail,test_size=0.3)

# Preparing email texts into word count matrix format 
Mails_bow = CountVectorizer(analyzer=split_into_words).fit(Mail.text)

# For all messages
all_emails_matrix = Mails_bow.transform(Mail.text)
all_emails_matrix.shape

# For training messages
train_emails_matrix = Mails_bow.transform(x_train.text)
train_emails_matrix.shape

# For testing messages
test_emails_matrix = Mails_bow.transform(y_test.text)
test_emails_matrix.shape 

### Without TFIDF matrices 
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes 99% Accuracy
classifier_mb = MB()
classifier_mb.fit(train_emails_matrix,x_train.type)
train_pred_m = classifier_mb.predict(train_emails_matrix)
accuracy_train_m = np.mean(train_pred_m==x_train.type)

test_pred_m = classifier_mb.predict(test_emails_matrix)
accuracy_test_m = np.mean(test_pred_m==y_test.type) #96% Accuracy


# Gaussian Naive Bayes 
classifier_gb = GB() #91% Accuracy
classifier_gb.fit(train_emails_matrix.toarray(),x_train.type.values) 
train_pred_g = classifier_gb.predict(train_emails_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==x_train.type)

test_pred_g = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==y_test.type) # 84%














