from mongoengine import *

class Tweets(Document):
    id = LongField(primary_key=True)

    parsed = BooleanField(required=True, default=False)
    dataset_name = StringField(required=True)
    label = StringField(required=True)
    parseable = BooleanField(required=True, default=False)

    user_name = StringField(required=False)
    tweet_body = StringField(required=False)

    meta = {
        'indexes': [
            'dataset_name',
            'parsed'
        ]
    }