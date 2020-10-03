#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install pandas')
get_ipython().system('pip3 install spotipy')
import pandas as pd 
import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials 
sp = spotipy.Spotify() 

cid ="e12f9ec40fe84d26b7b1755b1be0f3d7" 
secret = "de818023729647968d47bdfbdf607b3f" 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False


# In[ ]:


sp.user_playlist_tracks("31obxjrwpdtksnjtof7oopbelnwq", "7K3yIPMdOZcAtbev57z6ID")


# In[ ]:



def analyze_playlist(creator, playlist_id):
    
    # Create empty dataframe
    playlist_features_list = ["artist","album","track_name",  "track_id","danceability","energy","key","loudness","mode", "speechiness","instrumentalness","liveness","valence","tempo", "duration_ms","time_signature"]
    
    playlist_df = pd.DataFrame(columns = playlist_features_list)
    
    # Loop through every track in the playlist, extract features and append the features to the playlist df
    
    playlist = sp.user_playlist_tracks(creator, playlist_id)["items"]
    for track in playlist:
        # Create empty dict
        playlist_features = {}
        # Get metadata
        playlist_features["artist"] = track["track"]["album"]["artists"][0]["name"]
        playlist_features["album"] = track["track"]["album"]["name"]
        playlist_features["track_name"] = track["track"]["name"]
        playlist_features["track_id"] = track["track"]["id"]
        
        # Get audio features
        audio_features = sp.audio_features(playlist_features["track_id"])[0]
        for feature in playlist_features_list[4:]:
            playlist_features[feature] = audio_features[feature]
        
        # Concat the dfs
        track_df = pd.DataFrame(playlist_features, index = [0])
        playlist_df = pd.concat([playlist_df, track_df], ignore_index = True)
        
    return playlist_df


# In[ ]:


analyze_playlist("31obxjrwpdtksnjtof7oopbelnwq", "7K3yIPMdOZcAtbev57z6ID").to_csv("unLikedSongs.csv", index = False)


# In[ ]:





# In[ ]:




