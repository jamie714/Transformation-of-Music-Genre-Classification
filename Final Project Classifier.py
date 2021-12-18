#!/usr/bin/env python
# coding: utf-8

# In[89]:


# load in
import numpy as np
import nltk
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

old = np.loadtxt('lyrics_old.csv',dtype=str,delimiter=',',encoding='latin1')
new = np.loadtxt('lyrics_new.csv',dtype=str,delimiter=',',encoding='latin1')


# In[90]:


# prepare and separate data

# song[0] = index
# song[1] = song title
# song[2] = year
# song[3] = artist
# song[4] = genre
# song[5] = lyrics
# song[6] = word count

title_old = []
title_new = []
year_old = []
year_new = []
artist_old = []
artist_new = []
genre_old = []
genre_new = []
lyrics_old = []
lyrics_new = []

# pick out title, year, artist, genre and lyrics
for song in old:
    title_old.append(song[1])
    year_old.append(song[2])
    artist_old.append(song[3])
    genre_old.append(song[4])
    lyrics_old.append(song[5])
for song in new:
    title_new.append(song[1])
    year_new.append(song[2])
    artist_new.append(song[3])
    genre_new.append(song[4])
    lyrics_new.append(song[5])
# cut original category labels (first element)
del title_old[0]
del title_new[0]
del year_old[0]
del year_new[0]
del artist_old[0]
del artist_new[0]
del genre_old[0]
del genre_new[0]
del lyrics_old[0]
del lyrics_new[0]


# In[164]:


# shuffle lists and also test-train split old (without shuffle)
title_old, year_old, artist_old, genre_old, lyrics_old = shuffle(title_old, year_old,                                                                  artist_old, genre_old, lyrics_old)
title_new, year_new, artist_new, genre_new, lyrics_new = shuffle(title_new, year_new,                                                                  artist_new, genre_new, lyrics_new)

train_lyrics_old, test_lyrics_old, train_genre_old, test_genre_old = train_test_split(lyrics_old, genre_old,                                                                                       test_size=.2, shuffle=False)


# In[165]:


# logistic regression pipeline
pipe1 = None
pipe1 = Pipeline([('tfidf',TfidfVectorizer()),('clf',LogisticRegression())])
# fit to 'old' train data
pipe1.fit(train_lyrics_old,train_genre_old)
# predict and score for 'old' test data
predict_old=pipe1.predict(test_lyrics_old)
old_score=pipe1.score(test_lyrics_old,test_genre_old)
# predict and score for 'new' data
predict_new=pipe1.predict(lyrics_new)
new_score=pipe1.score(lyrics_new,genre_new)


# In[166]:


# regression results
print('Old Lyrics Test Set Accuracy: '+str(old_score))
print('New Lyrics Test Set Accuracy: '+str(new_score))


# In[153]:


# 5-neighbors pipeline
pipe2 = None
pipe2 = Pipeline([('tfidf',TfidfVectorizer()),('km',KMeans(n_clusters=5))])
# fit to 'old' train data
pipe2.fit(lyrics_old)
# get clusters
kmean = pipe2.named_steps.km
# labels: 0-4
labels=kmean.labels_


# In[154]:


# 5-neighbors results

# counters
jazzCountZero = 0
jazzCountOne = 0
jazzCountTwo = 0
jazzCountThree = 0
jazzCountFour = 0

rapCountZero = 0
rapCountOne = 0
rapCountTwo = 0
rapCountThree = 0
rapCountFour = 0

popCountZero = 0
popCountOne = 0
popCountTwo = 0
popCountThree = 0
popCountFour = 0

countryCountZero = 0
countryCountOne = 0
countryCountTwo = 0
countryCountThree = 0
countryCountFour = 0

rockCountZero = 0
rockCountOne = 0
rockCountTwo = 0
rockCountThree = 0
rockCountFour = 0

# iterate thru cluster labels
for i in range(len(labels)):
    label = labels[i]
    genre = genre_new[i]
    if genre == 'Jazz':
        if label == 0:
            jazzCountZero += 1
        elif label == 1:
            jazzCountOne += 1
        elif label == 2:
            jazzCountTwo += 1
        elif label == 3:
            jazzCountThree += 1
        elif label == 4:
            jazzCountFour += 1
    elif genre == 'Hip-Hop':
        if label == 0:
            rapCountZero += 1
        elif label == 1:
            rapCountOne += 1
        elif label == 2:
            rapCountTwo += 1
        elif label == 3:
            rapCountThree += 1
        elif label == 4:
            rapCountFour += 1
    elif genre == 'Pop':
        if label == 0:
            popCountZero += 1
        elif label == 1:
            popCountOne += 1
        elif label == 2:
            popCountTwo += 1
        elif label == 3:
            popCountThree += 1
        elif label == 4:
            popCountFour += 1
    elif genre == 'Country':
        if label == 0:
            countryCountZero += 1
        elif label == 1:
            countryCountOne += 1
        elif label == 2:
            countryCountTwo += 1
        elif label == 3:
            countryCountThree += 1
        elif label == 4:
            countryCountFour += 1
    elif genre == 'Rock':
        if label == 0:
            rockCountZero += 1
        elif label == 1:
            rockCountOne += 1
        elif label == 2:
            rockCountTwo += 1
        elif label == 3:
            rockCountThree += 1
        elif label == 4:
            rockCountFour += 1


# In[155]:


# display neighboring results
print('CLUSTER ZERO Counts:'+ ' Jazz: '+str(jazzCountZero)+', Hip-Hop: '+str(rapCountZero)+      ', Pop: '+str(popCountZero)+', Country: '+str(countryCountZero)+', Rock: '+str(rockCountZero))
print()
print('CLUSTER ONE Counts:'+' Jazz: '+str(jazzCountOne)+', Hip-Hop: '+str(rapCountOne)+      ', Pop: '+str(popCountOne)+', Country: '+str(countryCountOne)+', Rock: '+str(rockCountOne))
print()
print('CLUSTER TWO Counts:'+' Jazz: '+str(jazzCountTwo)+', Hip-Hop: '+str(rapCountTwo)+      ', Pop: '+str(popCountTwo)+', Country: '+str(countryCountTwo)+', Rock: '+str(rockCountTwo))
print()
print('CLUSTER THREE Counts:'+' Jazz: '+str(jazzCountThree)+', Hip-Hop: '+str(rapCountThree)+      ', Pop: '+str(popCountThree)+', Country: '+str(countryCountThree)+', Rock: '+str(rockCountThree))
print()
print('CLUSTER FOUR Counts:'+' Jazz: '+str(jazzCountFour)+', Hip-Hop: '+str(rapCountFour)+      ', Pop: '+str(popCountFour)+', Country: '+str(countryCountFour)+', Rock: '+str(rockCountFour))

