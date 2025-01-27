import pandas as pd
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from sklearn.decomposition import TruncatedSVD

try:
    ratings_df = pd.read_csv(r'data\ratings.csv')
    movies_df = pd.read_csv(r'data\movies.csv')
    print("Files opened successfully!!")
except FileNotFoundError:
    print("Dataset not found. Please try again")
except Exception:
    print("Error occurred. Please try again"); 

def dataset_check():
    pd.set_option('display.max_columns', None)  #to see columns & rows in entirety without truncation
    pd.set_option('display.max_rows', None)

    print("Printing information about the dataset")
    print(ratings_df.info())
    print(movies_df.info())

    print("Printing the first 15 rows of the data")
    print(ratings_df.head(15))
    print(movies_df.head(15))

    pd.reset_option('display.max_columns')  #resetting to default display settings
    pd.reset_option('display.max_rows')
#dataset_check()
#merging the datasets and printing the first 10 rows
merged_df= pd.merge(ratings_df, movies_df, on='movieId')
print(merged_df.head(10))

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

#function to recommend movies (creates a list of recommended movies)
def recommend_user_movies(user_Id,user_item_matrix, predicted_ratings, top_n):
    user_ratings=predicted_ratings[user_Id]
    top_n_movies=np.argsort(user_ratings)[-top_n:][::-1]

    recommend_movies=[]
    for i in top_n_movies:
        movie_id= user_item_matrix.columns[i]
        movie_data= movies_df[movies_df['movieId']==movie_id]
        if not movie_data.empty:
            title= movie_data['title'].values[0]
            genres= movie_data['genres'].values[0]
            predicted_ratings= round(user_ratings[i],1)
            recommend_movies.append([title, genres, predicted_ratings])

    return recommend_movies

def display_recommendation(recommended_movies):
    table= PrettyTable()
    table.field_names=['Movie Name/Title', 'Genres', 'Predicted Rating for You']
    for movie in recommended_movies:
        table.add_row(movie)
    
    print(table)

def sample_run():
    user_Id = 10
    top_n = 15
    recommended_movies = recommend_user_movies(user_Id, user_movie_matrix, predicted_ratings, top_n)
    display_recommendation(recommended_movies)
sample_run()
