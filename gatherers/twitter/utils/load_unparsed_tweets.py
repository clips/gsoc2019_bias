import pandas as pd

from gatherers.twitter.data.database_api import Mongo_API


def load_from_file(file_name):
    return pd.read_csv(file_name, delimiter=',')

def add_unparsed_tweets_to_database(file_name, dataset_name = None):
    if dataset_name is None: dataset_name = file_name
    tweets = load_from_file(file_name)
    for _, row in tweets.iterrows():
        Mongo_API().insert_unparsed_tweet(tweet_id=row['tweet_id'], label=row['label'], dataset_name=dataset_name)

if __name__ == "__main__":
    add_unparsed_tweets_to_database('hatespeech-labels.csv', "hatespeech-twitter-founta")