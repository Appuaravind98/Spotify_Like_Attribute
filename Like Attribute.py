#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install numpy')
import pandas as pd
import numpy as np


# In[ ]:


df1= pd.read_csv('LikedSongs.csv')
df2= pd.read_csv('unLikedSongs.csv')


# In[ ]:


df1['Target'] = np.ones((len(df1), 1), dtype=int)
df1.head()


# In[ ]:


df2['Target'] = np.zeros((len(df2), 1), dtype=int)
df2.head()


# df1.shape
# df2.shape

# In[ ]:


df1.shape


# In[ ]:


df2.shape


# In[ ]:


songs = df1.append(df2,ignore_index=False)
songs.tail()


# In[ ]:


songs.shape


# In[ ]:


songs.to_csv("Songs.csv", index = False)


# In[ ]:


songs = songs.drop_duplicates()


# In[ ]:


songs.shape


# In[ ]:


songs= songs.drop(['track_id', 'artist', 'album'] , axis= 1)


# In[ ]:


songs.head()


# In[ ]:


prediction = songs.drop(['track_name'], axis= 1)
prediction.head()


# In[ ]:


get_ipython().system('pip3 install matplotlib')
get_ipython().system('pip3 install seaborn')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(16,16))
plt.subplot(4,4,1)
sns.distplot(songs[songs['Target']==1]['danceability'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['danceability'], color='blue', bins=40)
plt.subplot(4,4,2)
sns.distplot(songs[songs['Target']==1]['energy'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['energy'], color='blue', bins=40)
plt.subplot(4,4,3)
sns.distplot(songs[songs['Target']==1]['key'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['key'], color='blue', bins=40)
plt.subplot(4,4,4)
sns.distplot(songs[songs['Target']==1]['loudness'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['loudness'], color='blue', bins=40)
plt.legend((1,0))


# In[ ]:


plt.figure(figsize=(16,16))
plt.subplot(4,4,1)
sns.distplot(songs[songs['Target']==1]['mode'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['mode'], color='blue', bins=40)
plt.subplot(4,4,2)
sns.distplot(songs[songs['Target']==1]['speechiness'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['speechiness'], color='blue', bins=40)
plt.subplot(4,4,3)
sns.distplot(songs[songs['Target']==1]['instrumentalness'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['instrumentalness'], color='blue', bins=40)
plt.subplot(4,4,4)
sns.distplot(songs[songs['Target']==1]['liveness'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['liveness'], color='blue', bins=40)
plt.legend((1,0))


# In[ ]:


plt.figure(figsize=(16,16))
plt.subplot(4,4,1)
sns.distplot(songs[songs['Target']==1]['valence'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['valence'], color='blue', bins=40)
plt.subplot(4,4,2)
sns.distplot(songs[songs['Target']==1]['tempo'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['tempo'], color='blue', bins=40)
plt.subplot(4,4,3)
sns.distplot(songs[songs['Target']==1]['duration_ms'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['duration_ms'], color='blue', bins=40)
plt.subplot(4,4,4)
sns.distplot(songs[songs['Target']==1]['time_signature'], color='red', bins=40)
sns.distplot(songs[songs['Target']==0]['time_signature'], color='blue', bins=40)
plt.legend((1,0))


# In[ ]:


X_train = prediction.drop('Target', axis=1)
X_test = songs.drop(['Target','track_name'], axis=1)
y_train = prediction['Target']
y_test = songs['Target']


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


y_train.tail()


# In[ ]:


y_test.head()


# In[ ]:


get_ipython().system('pip3 install sklearn')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


# In[ ]:


lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


# In[ ]:


lr_pred = lr_model.predict(X_test)
print(confusion_matrix(y_test, lr_pred))
print('\n')
print(classification_report(y_test, lr_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 


# In[ ]:


knn_model = KNeighborsClassifier() 


# In[ ]:


knn_model.fit(X_train, y_train) 


# In[ ]:


knn_pred = knn_model.predict(X_test)
print(confusion_matrix(y_test, knn_pred))
print('\n')
print(classification_report(y_test, knn_pred))


# In[ ]:


songs['prediction'] = knn_pred


# In[ ]:


songs.sort_values('track_name').head()


# In[ ]:


final_prediction = songs[['track_name','Target','prediction']]


# In[ ]:


final_prediction


# In[ ]:


final_prediction.to_csv("Songs_Like_prediction.csv", index = False)


# In[ ]:




