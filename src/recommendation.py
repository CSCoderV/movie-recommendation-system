#first importing libraries (using short forms/ psuedonames)

import panda as pd
df= pd.read_csv('data.csv')

#to see the first 10 rows of the data
print("Printing the first 10 rows of the data")
print(df.head(10))

#to check the general info about the dataset
print("Printing information about the dataset")
print(df.info())

#basic stats
print(df.describe())
