from mongoengine import connect

from gatherers.twitter.data.database_odm import Tweets

class Mongo_API:
    def __init__(self):
        db_name = 'Bias'
        connect(db_name)

    def insert_unparsed_tweet(self, tweet_id : int, dataset_name : str, label : str):
        tweet = Tweets(id = tweet_id, dataset_name = dataset_name, label = label)
        tweet.save()

    def insert_parse_for_tweet(self, tweet_id : int, tweet_body : str, user_name : str = None):
        pass

    def get_tweets(self, dataset_name : str = None, parsed : bool = True, limit : int = 100):
        if dataset_name is None:
            return Tweets.objects(parsed = parsed)[:limit]
        else:
            return Tweets.objects(dataset_name = dataset_name)