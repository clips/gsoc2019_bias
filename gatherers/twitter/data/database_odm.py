from mongoengine import *

class Tweets(Document):
    id = LongField(primary_key=True)
    user_name = StringField(required=False)
    tweet_content = StringField(required=False)
    parsed = BooleanField(required=True, default=False)
    dataset_name = StringField(required=True)
    label = StringField(required=True)

    meta = {
        'indexes': [
            'dataset_name',
            'parsed'
        ]
    }