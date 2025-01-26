import pandas as pd
import numpy as np
try:
    ratings_df = pd.read_csv(r'C:\Users\vedd1\movie-recommendation-system\data\ratings.csv')
    movies_df = pd.read_csv(r'C:\Users\vedd1\movie-recommendation-system\data\movies.csv')
    print("Files opened successfully!!")
except FileNotFoundError:
    print("Dataset not found. Please try again")
except Exception:
    print("Error occurred. Please try again"); 

print("Printing the first 15 rows of the data")
print(ratings_df.head(15))
print(movies_df.head(15))

print("Printing information about the dataset")
print(ratings_df.info())
print(movies_df.info())

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

