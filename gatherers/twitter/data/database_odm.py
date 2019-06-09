from mongoengine import *

class Tweets(Document):
    id = LongField(primary_key=True)

    dataset_name = StringField(required=True)
    label = StringField(required=True)
    parsed = BooleanField(required=True, default=False)
    parsable = BooleanField(required=True, default=False)

    user_name = StringField(required=False)
    tweet_body = StringField(required=False)
    date = DateTimeField(required=False)

    meta = {
        'indexes': [
            'dataset_name',
            'parsed'
        ]
    }