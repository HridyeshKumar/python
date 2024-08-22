#Movie Recommender System
'''Recommender Systems, also labelled as recommendation systems, are statistical algorithms that recommend products to users based on similarities between the buying trends of various user or similarities between the products.

Collaborative Filtering:- The process used to calculate similaritiies between the buying trends of various users or similarities between products is called collaborative filtering.

User based collaborative filtering:- If two user X and Y, like products A and B and there is another user Z who likes product A, then the product B will also be recommended to user Z.

Item-based collaborative filtering:- Inthis products are recommended based on similarities between themselves. For instance if a user likes product A and product A has properties X and Y will be recommended to the user.'''

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''The dataset contains around 100,000 movie reviews applied to 9,000 movies by 600 users.'''
movie_ids_titles=pd.read_csv("movies.csv")
movie_ids_titles.head()
movie_ids_titles.shape
movie_ids_ratings=pd.read_csv("ratings.csv")
movie_ids_ratings.head()
movie_ids_ratings.shape
'''Data Preprocessing:- We need a dataframe that consists of userId, movieId, title and ratings'''
movie_ids_titles.drop(['genres'],inplace=True,axis=1)
movie_ids_titles.head()
movie_ids_ratings.drop(["timestamp"],inplace=True,axis=1)
movie_ids_ratings.head()
merged_movie_df=pd.merge(movie_ids_ratings,movie_ids_titles,on='movieId')
merged_movie_df.head()
'''Data Visualization:- Let's first group the dataset by title and see what information we can get regarding the ratings of movies.'''
merged_movie_df.groupby('title').describe()
merged_movie_df.groupby('title')['rating'].mean().head()
'''Let's sort the movie titles by the descending order of the average user ratings'''
merged_movie_df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
'''Let's now print the movies in the descending order of their rating counts'''
merged_movie_df.groupby('title')['rating'].count().sort_values(ascending=False).head()
'''A movie which is rated by large number of people is usually a good movie.
Let's create a dataframe that shows the title, mean rating and the rating counts.'''
