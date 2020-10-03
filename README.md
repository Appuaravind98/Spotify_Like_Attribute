# Spotify_Like_Attribute
# dataSet.py 
 This file conaitns How you can extract data like Song attributes and track_name from  from Spotify libary  and save it into CSV format.
I created 2 spotify plalist 1 containing  almost 75% of songs that I hear frequently.the other contains songs that I hear rarely.After saving  both Playlist as csv format. In this Article i have Explained how I created a Dataset using Spotipy Library https://medium.com/@appuaravind/creating-dataset-for-your-spotify-library-582a89babce3
# LIkeattribute.py
I added an extra column "Target" to the first CSV file and gave the value 1.This means they belong to my favorite songs category.After that I added a column "Target" to Second CSV file gave the value 0. then I combined these to CSV in to one by appending.I removed all unwanted data from the dataframe like albumn, Track_id etc. I checked with both logistic regression Model and  K-Nearest Neighbor Machine Learning Model to predict.My K-Nearest Neighbor Machine Learning Model
had a better precision with 75% so i decided to go for it  and  predicted the Song like of my choice with that 
# I advice you to use jupyter notebooks to do this one So you can run each and every step one after another   
