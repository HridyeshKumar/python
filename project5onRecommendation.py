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
movie_rating_mean_count=pd.DataFrame(columns=['rating_mean','rating_count'])
movie_rating_mean_count["rating_mean"]=merged_movie_df.groupby('title')['rating'].mean()
movie_rating_mean_count["rating_count"]=merged_movie_df.groupby('title')['rating'].count()
movie_rating_mean_count.head()
'''The above dataframe contains movie title, average rating (ratings mean) and the number of rating_counts
We will plot a histogram to see how the average ratings are distributed. '''
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
movie_rating_mean_count['rating_mean'].hist(bins=30,color='purple')
#Distribution for rating counts
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
movie_rating_mean_count['rating_count'].hist(bins=33,color='green')
'''There are around 7000 movies with less than 10 rating counts. The number of movies decrease with an increase in ratings counts. Movies with more than 50 rating are very few.
It is also interesting to see the relationship between mean ratings and rating counts of a movie.'''
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.regplot(x="rating_mean",y="rating_count",data=movie_rating_mean_count,color="brown")
'''From the above graph in the top right portion,you can see that the movies with a higher number of rating counts tend to have higher mean ratings as well.
Let's sort our dataset by rating counts and see the average ratings of the movies with the top 5 highest number of ratings.'''
movie_rating_mean_count.sort_values("rating_count",ascending=False).head()

#Item Based Collaborative Filtering
'''In item based collaborative filtering, products are recommended based on common characteristics.
The first step is to create a dataframe where each movie is represented by a column and rows contain user ratings for movies.'''
user_movie_rating_matrix=merged_movie_df.pivot_table(index="userId",columns="title",values="rating")
print(user_movie_rating_matrix)
user_movie_rating_matrix.shape
'''The Dataset contains 610 unique users and 9719 unique movies.
Now we will find the movie recommendation based on a single movie and then based on multiple movies.
Finding recommendations based on a single movie. Suppose we want to find the recommendation based on the movie Pulp Fiction.
First we will filter the column that contains the user ratings for the movie.'''
pulp_fiction_ratings=user_movie_rating_matrix["Pulp Fiction (1994)"]
'''Next, we will find the correlation between the user ratings of all the movies and the user ratings for the movie pulp fiction'''
pulp_fiction_correlations=pd.DataFrame(user_movie_rating_matrix.corrwith(pulp_fiction_ratings,columns=["pf_corr"])
pulp_fiction_correlations.sort_values("pf_corr",ascending=False).head(5)
'''Correlation itself is not giving meaningful results, one solution to this problem can be that in addition to the correlation between the movies, we also use rating counts, for the correlated movie as a criteria for finding the best revommendation.'''
pulp_fiction_correlations=pulp_fiction_correlations.join(movie_rating_mean_count["rating_count"])
pulp_fiction_correlations.head()
'''The pf_corr column contains some NaN values. This is because there can be movies that are rated by users who did not rate Pulp Fiction (1994). In such cases, correlation will be null.
We will remove all the movies with null correlation with Pulp Fiction (1994).'''
pulp_fiction_correlations.dropna(inplace=True)
pulp_fiction_correlations.sort_values("pf_corr",ascending=False).head()
'''A better way is to find the movies with the rating counts of atleast 50 and having the highest correlation with Pulp Fiction (1994).'''
pulp_fiction_correlations_50=pulp_fiction_correlations[pulp_fiction_correlations['rating_count']>50]
pulp_fiction_correlation_50.sort_values("pf_corr",ascending=False).head()
'''Finding the recommendation based on multiple movies. The first step is to create a dataframe, which contains a correlation between all the movies in our dataset in the form of a matrix.'''
all_movie_correlations=user_movie_rating_matrix.corr(method="pearson",min_periods=50)
all_movie_correlations.head()
'''Now suppose a new user logs into the website. The user has already watched three movies and has given ratings to those movies.'''
movie_data=[['Forrest Gump (1994)',4.0],['Fight Club (1999)',3.5],['Interstellar (2014)',4.0]]
test_movies=pd.DataFrame(movie_data,columns=['Movie_Name','Movie_Rating'])
test_movies.head()
'''We will be recommending movies from our dataset based on the ratings by a new user for these three movies.'''
print(test_movies['Movie_Name'][0])
print(test_movies['Movie_Rating'][0])
'''From all_movie_correlations dataframe, let's obtain correlation values for the movies related to Forrest Gump (1994)'''
all_movie_correlations['Forrest Gump (1994)'].dropna()
'''Next, we will iterate through the three movies in the test_movies dataframe, find the correlated movies, and then multiply the correlation of all the correlated movies with the ratings of the input movie.
The correlated movies, along with the weighted correlation are appended to an empty series named recommended movies.'''
recommended_movies=pd.Series()
for i in range(0,2):
   movie=all_movie_correlations[test_movies['Movie_Name'][i]].dropna()
   movie=movie.map(lambda movie_corr:movie_corr*test_movies["Movie_Rating"][i])
   recommended_movies=recommended_movies.append(movie)
print(recommended_movies)
'''To get a final recommendation, you can sort the movies in the descending order of the weighted correlation'''
recommended_movies.sort_values(inplace=True,ascending=False)
print(recommended_movies.head(10))
