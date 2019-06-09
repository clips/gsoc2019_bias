from datetime import datetime

from mongoengine import connect

from gatherers.twitter.data.database_odm import Tweets
from gatherers.twitter.utils.singleton import Singleton

class Mongo_API(metaclass=Singleton):
    def __init__(self):
        db_name = 'Bias'
        connect(db_name)

    def insert_unparsed_tweet(self, tweet_id : int, dataset_name : str, label : str):
        tweet = Tweets(id = tweet_id, dataset_name = dataset_name, label = label)
        tweet.save()

    def insert_parse_for_tweet(self, tweet_id : int, tweet_body : str, user_name : str = None, date : datetime = None):
        Tweets.objects(id = tweet_id).update_one(set__tweet_body = tweet_body)
        if(user_name is not None):
            Tweets.objects(id=tweet_id).update_one(set__user_name=user_name)
        if(date is not None):
            Tweets.objects(id=tweet_id).update_one(set__date=date)

    def get_tweets(self, dataset_name : str = None, limit : int = 100):
        if dataset_name is None:
            return Tweets.objects[:limit]
        else:
            return Tweets.objects(dataset_name = dataset_name)

    def get_parsable_tweet_id(self, dataset_name: str = None, parsed: bool = False):
        if dataset_name is None:
            return Tweets.objects(parsed=parsed).only('id')[:1]
        else:
            return Tweets.objects(dataset_name=dataset_name)

    def mark_unparseable(self, tweet_id : int):
        Tweets.objects(id = tweet_id).update_one(set__parsable = False)