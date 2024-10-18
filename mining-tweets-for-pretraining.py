import json
import pandas as pd
import spacy_fastlang
import spacy
import io


#Import data
data1 = pd.read_json("twitter-search-puta6", lines=True)
data2 = pd.read_json("twitter-search-puta7", lines=True)
#data3 = pd.read_json("twitter-search-pendeja", lines=True)
#data4 = pd.read_json("twitter-search-pendeja2", lines=True)
#data5 = pd.read_json("./twitter-search-pendeja", lines=True)
#data6 = pd.read_json("./twitter-search-puta-new", lines=True)
#data7 = pd.read_json("./twitter-search-putas", lines=True)
data = pd.concat([data1,data2])
data = data[['content']]
#data = data1[['content']]
data['content'] = data['content'].astype(str)
print("length unprocessed data puta:", len(data))

#Dropping ALL duplicate values
data.drop_duplicates()
#Dropping ALL null values
data = data.dropna()

#Print data length
print("length of dataframe: ", len(data))

# language detector
nlp = spacy.load('es_core_news_sm')
nlp.add_pipe('language_detector')

data['lang'] = data['content'].apply(lambda x: nlp(x)._.language)

es_tweets = data[data['lang'] == 'es']
es_tweets = es_tweets[['content']]

print("length processed tweets:", len(es_tweets))

#tweets = pd.read_csv("./tweets/puta-coño-estupida")
#tweets = tweets[['content']]

#new_tweets = pd.concat([es_tweets, tweets])
#new_tweets.drop_duplicates()
#new_tweets = new_tweets.dropna()

#print("all tweets", len(new_tweets))

es_tweets.to_csv("./tweets/new-puta-tweets")
