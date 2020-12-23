# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:35:24 2020

@author: axays
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import seaborn as sns

mails=pd.read_csv('C:\\Users\\axays\\Downloads\\emails')

mails.shape
mails.columns
mails.drop_duplicates(inplace=True)
mails.shape
mails.isnull().sum()

mails["Class"].value_counts()

#Non Abusive    44666
#Abusive         3410

sns.countplot(mails['Class'])


#Download the STOPWORDS
nltk.download('stopwords')

def process_text(content):
    #1st Remove PUNCTUATIONS
    #2nd Remove STOPWORDS
    #3rd Return alist of clean text words.
    
    #1st
    nonpunc=[char for char in content if char not in string.punctuation]
    nonpunc=''.join(nonpunc)
    
    #2nd
    clean_words=[word for word in nonpunc.split() if word.lower() not in stopwords.words('english')]
    
    #3rd
    return clean_words

#Show tokenization also called Lemmas.
mails['content'].head().apply(process_text)

#Example

Message1='Repetition makes reputation and reputation makes customers'
Message2='change begets change as much as repetition reinforces repetition'
print(Message1)
print()

from sklearn.feature_extraction.text import  CountVectorizer
Mod=CountVectorizer(analyzer=process_text).fit_transform([[Message1],[Message2]])
print(Mod)

#Convert a collection of contents to a matrix of tokens.
Convert=CountVectorizer(analyzer=process_text).fit_transform(mails['content'])

#Splitiing data into 80% Training and 20% Testing.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(Convert,mails['Class'],test_size=0.20,random_state=0)

Convert.shape

#Create and train Naive bayes classifier.
from sklearn.naive_bayes import MultinomialNB
Classifier=MultinomialNB().fit(X_train,y_train)


#Print the predictions.
print(Classifier.predict(X_train))


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
pred=Classifier.predict(X_train)
print(classification_report(y_train,pred))

#Print the Actual values.
print(y_train.values)

# Creating a WordCloud
from wordcloud import WordCloud
Allwords=' '.join([word for word in mails['content']])
Cloud=WordCloud(width=500,height=300,random_state=21,max_font_size=100).generate(Allwords)


plt.imshow(Cloud,interpolation='bilinear')
plt.axis('off')
plt.show()
