#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk

data = pd.read_csv("C:/Users/Fast-Pwr/Desktop/covid19_tweets.csv")
print(data)


# In[13]:


nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["text"] = data["text"].apply(clean)


# In[14]:


data["text"]


# In[15]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["text"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["text"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["text"]]


# In[16]:


data = data[["text", "Positive", 
             "Negative", "Neutral"]]
print(data)


# In[17]:


x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive 😊 ")
    elif (b>a) and (b>c):
        print("Negative 😠 ")
    else:
        print("Neutral 🙂 ")
sentiment_score(x, y, z)


# In[18]:


print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)


# In[19]:


data.info()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sn
data.hist(bins=18, figsize=(20, 20))
plt.tight_layout()


# In[21]:


data.to_csv('C:/Users/Fast-Pwr/Desktop/covid19_tweets_sentiment_done.csv')


# In[22]:


data = pd.read_csv("C:/Users/Fast-Pwr/Desktop/covid19_tweets_sentiment_done.csv")
print(data)


# In[24]:


data= pd.read_csv("C:/Users/Fast-Pwr/Desktop/covid19_tweets_sentiment_done.csv")
data


# In[ ]:




