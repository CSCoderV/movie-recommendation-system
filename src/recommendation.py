import pandas as pd
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from sklearn.decomposition import TruncatedSVD

#the function helps to load the dataset. Make sure you have downloaded the dataset :)
def load_data():
    try:
        ratings_df = pd.read_csv(r'data\ratings.csv')
        movies_df = pd.read_csv(r'data\movies.csv')
        print("Files opened successfully!!")
        return ratings_df, movies_df
    except FileNotFoundError:
        print("Dataset not found. Please try again")
    except Exception:
        print("Error occurred. Please try again");
        return (None, None)
    
        return (None, None)

def dataset_check(ratings_df, movies_df):
    pd.set_option('display.max_columns', None)  #seeing columns & rows entirely without truncation just to get a sense of it
    pd.set_option('display.max_rows', None)

    print("Printing information about the dataset")
    print(ratings_df.info())
    print(movies_df.info())

    print("Printing the first 15 rows of the data")
    print(ratings_df.head(15))
    print(movies_df.head(15))

    pd.reset_option('display.max_columns')  #resetting to default display settings
    pd.reset_option('display.max_rows')

def merge_data(ratings_df, movies_df):
    # the function helps in merging the datasets and printing the first few rows (I have set it to 10 but you can change the number)
    merged_df= pd.merge(ratings_df, movies_df, on='movieId')
    #print(merged_df.head(10))

    #print("Check for missing values")
    #print(merged_df.isnull().sum())
    #print("Dropping the missing values")
    merged_df= merged_df.dropna()

    print("Checking for duplicates")
    print(merged_df.duplicated().sum())

    #dropping duplicates (to ensure the accracy of the data otherwise it might affect the accuracy of the model)
    merged_df= merged_df.drop_duplicates()

    return merged_df.dropna().drop_duplicates()

def create_user_item_matrix(merged_df):
    #creating a user-item matrix 
    user_movie_matrix=merged_df.pivot(index='userId', columns='movieId', values='rating');

    #print("User-Movie Matrix")
    #print(user_movie_matrix.head(15))
    user_movie_matrix= user_movie_matrix.fillna(0)
    return user_movie_matrix

def run_recommendations(user_movie_matrix):
    svd= TruncatedSVD(n_components=30)      #Decomposing the matrix into a 30*30 matrix to hopefully reduce the noise from data
    user_factors=svd.fit_transform(user_movie_matrix)   #Decomposed user factor
    item_factors=svd.components_    #Decomposed item factors
    print(np.sum(svd.explained_variance_ratio_))

    #Reconstructing the matrix
    reconstructed_matrix= np.dot(user_factors,item_factors)
    #print("Reconstructed Matrix: ")
    #print(reconstructed_matrix[:3,:3])

    #predicting the ratings users might give to the movies they haven't watched
    predicted_ratings=np.dot(user_factors, item_factors)
    return predicted_ratings

#function to recommend users movies ( basically, it creates a list of recommended movies)
def recommend_user_movies(user_Id,user_item_matrix, predicted_ratings, top_n,movies_df):
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

def sample_run(user_movie_matrix, predicted_ratings, movies_df):
    user_Id = 7
    top_n = 10
    recommended_movies = recommend_user_movies(user_Id, user_movie_matrix, predicted_ratings, top_n,movies_df)
    display_recommendation(recommended_movies)

def user_creation():
    print("Hello there! Welcome to the Movie Recommendation System")
    user_input1=eval(input("Are you a new user? (1) Yes or (2) No"))
    if user_input1==1 or userinput=='Yes' or user_input1=='yes' or user_input1=='y':
        print("Please wait a moment while we create your profile")
        user_id= print ("Please enter the user ID you want: ")
        if userid in user_movie_matrix.index:
            print("User ID already exists. Please try again")
            user_creation()
        else:
            print("User ID created successfully")
        print("Now please select the genres of movies yo like to watch (max of 3)")
        genres_in_dataset = movies_df['genres'].str.split('|').explode().unique()
        print("The following are the genres of movies in the dataset:")
        table1 = PrettyTable()
        table1.field_names = ['No.', 'Genres']
        for i, genre in enumerate(genres_in_dataset, start=1):
            table1.add_row([i, genre])
        print(table1)





def main():

    ratings_df, movies_df = load_data()
    if ratings_df is None or movies_df is None:
        return

    merged_df = merge_data(ratings_df, movies_df)
    user_movie_matrix = create_user_item_matrix(merged_df)
    predicted_ratings = run_recommendations(user_movie_matrix)
    sample_run(user_movie_matrix, predicted_ratings, movies_df)
main()
