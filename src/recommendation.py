import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

try:
    ratings_df = pd.read_csv(r'C:\Users\vedd1\movie-recommendation-system\data\ratings.csv')
    movies_df = pd.read_csv(r'C:\Users\vedd1\movie-recommendation-system\data\movies.csv')
    print("Files opened successfully!!")
except FileNotFoundError:
    print("Dataset not found. Please try again")
except Exception:
    print("Error occurred. Please try again"); 

print("Printing information about the dataset")
print(ratings_df.info())
print(movies_df.info())

print("Printing the first 15 rows of the data")
print(ratings_df.head(15))
print(movies_df.head(15))


pd.set_option('display.max_columns', None)  #to see columns & rows in entirety without truncation
pd.set_option('display.max_rows', None)

#merging the datasets and printing the first 10 rows
merged_df= pd.merge(ratings_df, movies_df, on='movieId')
print(merged_df.head(10))
pd.reset_option('display.max_columns')  #resetting to default display settings
pd.reset_option('display.max_rows')

print("Check for missing values")
print(merged_df.isnull().sum())
print("Dropping the missing values")
merged_df= merged_df.dropna()

print("Checking for duplicates")
print(merged_df.duplicated().sum())

#dropping duplicates
merged_df= merged_df.drop_duplicates()

#creating a user-item matrix 
user_movie_matrix=merged_df.pivot(index='userId', columns='movieId', values='rating');

print("User-Movie Matrix")
print(user_movie_matrix.head(15))
user_movie_matrix= user_movie_matrix.fillna(0)

svd= TruncatedSVD(n_components=30)
user_factors=svd.fit_transform(user_movie_matrix)   #Decomposed user factor
item_factors=svd.components_    #Decomposed item factors
print(np.sum(svd.explained_variance_ratio_))

#Reconstructing the matrix
reconstructed_matrix= np.dot(user_factors,item_factors)
print("Reconstructed Matrix: ")
print(reconstructed_matrix[:3,:3])

#predicting the ratings users might give to the movies they haven't watched
predicted_ratings=np.dot(user_factors, item_factors)

def recommend_user_movies(user_Id,user_item_matrix, predicted_ratings, top_n=10):
    user_ratings=predicted_ratings[user_Id]
    top_n_movies=np.argsort(user_ratings)[-top_n:][::-1]

    print("Top 10 recommended movies for you: ")
    return top_n_movies
print(recommend_user_movies(10,user_movie_matrix,predicted_ratings))