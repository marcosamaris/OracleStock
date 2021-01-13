import sys

import numpy as np
import pandas as pd
import constants_stocks as cs

import constants_tweet as ct

from Tweet import Tweet
import tweepy
from textblob import TextBlob

import pickle
import nltk

def retrieving_tweets_polarity(symbol):   
    
#     nltk.download('punkt')
    auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
    auth.set_access_token(ct.access_token, ct.access_token_secret)
    user = tweepy.API(auth, wait_on_rate_limit=True)

    tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='pt').items(ct.num_of_tweets)

    tweet_list = []
    global_polarity = 0
    for tweet in tweets:
        tw = tweet.full_text
        blob = TextBlob(tw)
        polarity = 0
        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity
            global_polarity += sentence.sentiment.polarity
        tweet_list.append(Tweet(tw, polarity))

    if len(tweet_list) > 0:
        global_polarity = global_polarity / len(tweet_list)
    return global_polarity


all_polarity = []
ML_Techniques = ['conv', 'LSTM', 'BidirectionalLSTM', 'convLSTM1D', 'convLSTM2D']     


for stock in cs.stocks_codigo[int(sys.argv[1]):int(sys.argv[1]) + len(cs.stocks_codigo)]:
    
    polarity = retrieving_tweets_polarity(stock)
    print(stock, polarity)
    
    all_polarity.append([stock, polarity])
    
dataFrame = pd.DataFrame(all_polarity)

dataFrame.columns = ['Stock', 'Polarity']          

dataFrame.to_csv('logs/data-polarity.csv')

