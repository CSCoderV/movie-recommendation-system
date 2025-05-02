import pandas as pd
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from sklearn.decomposition import TruncatedSVD
import sqlite3

#the function helps to load the dataset. Make sure you have downloaded the dataset :)
def load_data():
    try:
        # Fix path separators for cross-platform compatibility
        ratings_df = pd.read_csv('data/ratings.csv')
        movies_df = pd.read_csv('data/movies.csv')
        print("Files opened successfully!!")
        return ratings_df, movies_df
    except FileNotFoundError:
        print("Dataset not found. Please try again")
        return None, None
    except Exception as e:
        print(f"Error occurred: {e}. Please try again");
        return None, None

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

    #print("Checking for duplicates")
    #print(merged_df.duplicated().sum())

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

def recommend_user_movies(user_Id,user_item_matrix, predicted_ratings, top_n,movies_df):
    #function to recommend users movies ( basically, it creates a list of recommended movies)
    if user_Id < 0 or user_Id >= predicted_ratings.shape[0]:
        print("Error while recommending!! Invalid user ID. Please try again.")
        return []
    user_ratings=predicted_ratings[user_Id]
    top_n_movies=np.argsort(user_ratings)[-top_n:][::-1]
    #empty list to store recommendations
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
    #creates table using PrettyTable library, adds movies to it and prints it
    if not recommended_movies:
        print("No movies to display.")
        return
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

def main():
    ratings_df, movies_df = load_data()
    if ratings_df is None or movies_df is None:
        return

    # Create or open SQLite database
    conn = sqlite3.connect('data/users.db')
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY)''')
    conn.commit()
    
    # Get the list of existing users from the database
    cursor.execute("SELECT user_id FROM users")
    existing_user_ids = [row[0] for row in cursor.fetchall()]
    
    # Process data
    merged_df = merge_data(ratings_df, movies_df)
    user_movie_matrix = create_user_item_matrix(merged_df)
    
    # Make sure all database users exist in the matrix
    for user_id in existing_user_ids:
        if user_id not in user_movie_matrix.index:
            user_movie_matrix.loc[user_id] = user_movie_matrix.mean()
    
    # Sort the matrix for consistency
    user_movie_matrix = user_movie_matrix.sort_index()
    
    # Run recommendations
    predicted_ratings = run_recommendations(user_movie_matrix)
    
    # Start user interaction
    user_interaction(user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
    
    conn.close()

def user_interaction(user_movie_matrix, predicted_ratings, movies_df, conn, cursor):
    print("Hello there! Welcome to the Movie Recommendation System")
    user_input1 = input("Are you a new user? (yes/no): ").lower()

    if user_input1 in ["yes", "y", "1"]:
        create_new_user(user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
    else:
        login_existing_user(user_movie_matrix, predicted_ratings, movies_df, conn, cursor)

def create_new_user(user_movie_matrix, predicted_ratings, movies_df, conn, cursor):
    print("Please wait a moment while we create your profile")
    
    try:
        user_id = int(input("Please enter a numeric user ID you want: "))
    except ValueError:
        print("Invalid input. Please enter a numeric ID.")
        create_new_user(user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        return
    
    # Check if user exists in database
    cursor.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
    if cursor.fetchone():
        print(f"User ID {user_id} already exists in our database. Please try again.")
        create_new_user(user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        return
    
    print(f"User ID {user_id} created successfully!")
    
    # Add user to matrix and database
    user_movie_matrix.loc[user_id] = user_movie_matrix.mean()
    user_movie_matrix = user_movie_matrix.sort_index()
    cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
    conn.commit()
    
    # Recalculate predictions with new user
    predicted_ratings = run_recommendations(user_movie_matrix)
    
    # Continue with genre selection
    setup_user_preferences(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)

def setup_user_preferences(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor):
    genres_in_dataset = list(movies_df['genres'].str.split('|').explode().unique())
    print("The following are the genres of movies in the dataset:")
    table1 = PrettyTable()
    table1.field_names = ['No.', 'Genres']
    for i, genre in enumerate(genres_in_dataset, start=1):
        table1.add_row([i, genre])
    print(table1)

    try:
        user_genres = [int(x.strip()) for x in input("Now please select the genres of movies you like to watch (max of 3), separated by commas: ").split(',')]
        
        if len(user_genres) > 3:
            print("Please select maximum of 3 genres")
            setup_user_preferences(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
            return

        selected_genres = [genres_in_dataset[i - 1] for i in user_genres if 1 <= i <= len(genres_in_dataset)]

        if not selected_genres:
            print("No valid genres selected. Using all genres by default.")
        else:
            print("Thank you for selecting your genres.")

        movies_in_genre = movies_df[movies_df['genres'].apply(lambda x: any(genre in x for genre in selected_genres))]

        if movies_in_genre.empty:
            print("Sorry, that seems to be an unpopular genre! No movies found for the selected genres. Please try again.")
            setup_user_preferences(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
            return

        displayed_movies = 0
        table2 = PrettyTable()
        table2.field_names = ['No.', 'Movie Name/Title']

        for i, movie in enumerate(movies_in_genre['title'].head(10), start=1):
            table2.add_row([i, movie])

        print(table2)

        user_movies = [int(x.strip()) for x in input("Now please select the movies you like to watch (max of 5), separated by commas: ").split(',')]
        if len(user_movies) > 5:
            print("Sorry. Please retry again (Error: Maximum of 5 movies can be selected)")
            setup_user_preferences(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        else:
            print("Thank you for selecting the movies you like to watch. Now, we will recommend movies for you. Please wait a moment.")
            user_index = user_movie_matrix.index.get_loc(user_id)
            recommended_movies = recommend_user_movies(user_index, user_movie_matrix, predicted_ratings, 10, movies_df)
            display_recommendation(recommended_movies)
            
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        setup_user_preferences(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        return

def login_existing_user(user_movie_matrix, predicted_ratings, movies_df, conn, cursor):
    try:
        user_id = int(input("Please enter your user ID: "))
    except ValueError:
        print("Invalid input. Please enter a numeric ID.")
        login_existing_user(user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        return
    
    # Check if user exists in database
    cursor.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
    if not cursor.fetchone():
        print(f"User ID {user_id} does not exist in our database. Please try again or create a new user.")
        user_interaction(user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        return
    
    # Add to matrix if not present but in database
    if user_id not in user_movie_matrix.index:
        user_movie_matrix.loc[user_id] = user_movie_matrix.mean()
        user_movie_matrix = user_movie_matrix.sort_index()
        predicted_ratings = run_recommendations(user_movie_matrix)
    
    print(f"Welcome back, user {user_id}!")
    show_user_menu(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)

def show_user_menu(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor):
    print("\nWhat would you like to do?")
    print("1. Get movie recommendations")
    print("2. Search for a specific movie")
    print("3. Exit")
    
    try:
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            user_index = user_movie_matrix.index.get_loc(user_id)
            recommended_movies = recommend_user_movies(user_index, user_movie_matrix, predicted_ratings, 10, movies_df)
            display_recommendation(recommended_movies)
            # Ask if they want to do something else
            if input("\nWould you like to do something else? (yes/no): ").lower() in ["yes", "y", "1"]:
                show_user_menu(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        elif choice == "2":
            search_term = input("Enter a movie title to search for: ")
            search_results = search_movies(movies_df, search_term)
            display_search_results(search_results)
            # Ask if they want to do something else
            if input("\nWould you like to do something else? (yes/no): ").lower() in ["yes", "y", "1"]:
                show_user_menu(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
        elif choice == "3":
            print("Thank you for using the Movie Recommendation System. Goodbye!")
            return
        else:
            print("Invalid choice. Please try again.")
            show_user_menu(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)
    except ValueError:
        print("Invalid input. Please try again.")
        show_user_menu(user_id, user_movie_matrix, predicted_ratings, movies_df, conn, cursor)

def search_movies(movies_df, search_term):
    """
    Search for movies by title (case-insensitive partial match)
    
    Args:
        movies_df: DataFrame containing movies data
        search_term: String to search for in movie titles
        
    Returns:
        DataFrame with matching movies
    """
    matches = movies_df[movies_df['title'].str.contains(search_term, case=False)]
    return matches.head(10)  # Limit to 10 results to avoid overwhelming output

def display_search_results(search_results):
    """
    Display search results in a formatted table
    
    Args:
        search_results: DataFrame with search results
    """
    if search_results.empty:
        print("No movies found matching your search term.")
        return
        
    table = PrettyTable()
    table.field_names = ['Movie ID', 'Movie Title', 'Genres']
    
    for _, movie in search_results.iterrows():
        table.add_row([movie['movieId'], movie['title'], movie['genres']])
    
    print(table)

main()
