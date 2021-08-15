# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:53:31 2020

@author: axays
"""

import pandas as pd
import numpy as np
import seaborn as sns
Books=pd.read_csv('C:\\Users\\axays\\Desktop\\Assignments\\book (2).csv',encoding='latin1')

Books.head()
Books.nunique()
Books.isnull().sum()
Books.columns


sns.countplot(x='Book.Rating',hue='User.ID',data=Books)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(Books['Book.Title'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

book_index = pd.Series(Books.index,index=Books['Book.Title']).drop_duplicates()


book_index['Decision in Normandy']

def get_book_recommendations(Name,topN):
    #topN = 10
    # Getting the book index using its title 
    book_id = book_index[Name]
    # Getting the pair wise similarity score for all the Books's with that 
    # books.
    cosine_scores = list(enumerate(cosine_sim_matrix[book_id]))
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    # Get the scores of top 10 most similar book's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    # Getting the book index 
    book_idx  =  [i[0] for i in cosine_scores_10]
    book_scores =  [i[1] for i in cosine_scores_10]
    # Similar books and scores
    Book_Ratings = pd.DataFrame(columns=["Book.Title","Book.Rating"])
    Book_Ratings["Book.Title"] = Books.loc[book_idx,"Book.Title"]
    Book_Ratings["Book.Rating"] = book_scores
    Book_Ratings.reset_index(inplace=True)  
    Book_Ratings.drop(["index"],axis=1,inplace=True)
    print (Book_Ratings)
    #return (anime_similar_show)
    
get_book_recommendations("Classical Mythology",topN=5)



