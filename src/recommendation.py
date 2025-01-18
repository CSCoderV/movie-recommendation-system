import pandas as pd
ratings_df = pd.read_csv(r'C:\Users\vedd1\movie-recommendation-system\data\ratings.csv')
movies_df = pd.read_csv(r'C:\Users\vedd1\movie-recommendation-system\data\movies.csv')
print("Printing the first 10 rows of the data")
print(ratings_df.head(10))
print(movies_df.head(10))

print("Printing information about the dataset")
print(ratings_df.info())
print(movies_df.info())

#merging the datasets
merged_df= pd.merge(ratings_df, movies_df, on='moviedId')
printf(merged_df.head(15))