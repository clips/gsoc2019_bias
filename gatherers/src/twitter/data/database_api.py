from datetime import datetime

from mongoengine import connect

from gatherers.src.twitter import Tweets
from gatherers.src.twitter import Singleton

# Singleton class for accessing the tweets database, connects to localhost MongoDB instance without any authentication
class Mongo_API(metaclass=Singleton):
    # Init block to connect to the database, as the class is a singleton this is called only once
    def __init__(self):
        db_name = 'Bias'
        connect(db_name)

    # Insert a tweet, marking it as parsable and unparsed, the tweet is assigned to a dataset and given a text label
    def insert_unparsed_tweet(self, tweet_id : int, dataset_name : str, label : str):
        tweet = Tweets(id = tweet_id, dataset_name = dataset_name, label = label)
        tweet.save()

    # Update a tweet to include the body, username and date
    def insert_parse_for_tweet(self, tweet_id : int, tweet_body : str, user_name : str = None, date : datetime = None):
        Tweets.objects(id = tweet_id).update_one(set__tweet_body = tweet_body)
        if(user_name is not None):
            Tweets.objects(id=tweet_id).update_one(set__user_name=user_name)
        if(date is not None):
            Tweets.objects(id=tweet_id).update_one(set__date=date)
        Tweets.objects(id = tweet_id).update_one(set__parsed = True)

    # Get tweets from a name dataset, up to limit
    def get_tweets(self, dataset_name : str = None, limit : int = 100):
        if dataset_name is None:
            return Tweets.objects[:limit]
        else:
            return Tweets.objects(dataset_name = dataset_name)

    # Get ids for tweets that can be parsed and haven't been parsed yet
    def get_parsable_tweet_ids(self, dataset_name: str = None, parsed: bool = False, skip : int = 0, limit : int = 100):
        if dataset_name is None:
            return Tweets.objects(parsed=parsed, parsable=True).scalar('id')[skip:skip+limit]
        else:
            return Tweets.objects(dataset_name=dataset_name, parsed=parsed, parsable=True).scalar('id')[skip:skip+limit]

    # Mark a given tweet as unparsable for the case they're unavailable from the Twitter side
    def mark_unparseable(self, tweet_id : int):
        Tweets.objects(id = tweet_id).update_one(set__parsable = False)

    # Count the total parsed tweets
    def count_total_tweets_processed(self):
        return Tweets.objects(parsed=True).count() + Tweets.objects(parsable=False).count()