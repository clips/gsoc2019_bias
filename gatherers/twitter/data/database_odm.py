from mongoengine import *

# The tweet object document mapping, contains both content fields (Tweet body, username, date) as well as metadata such
# as in which dataset it was contained, what labels it has and its own parsing status
class Tweets(Document):
    id = LongField(primary_key=True)

    dataset_name = StringField(required=True)
    label = StringField(required=True)
    parsed = BooleanField(required=True, default=False)
    parsable = BooleanField(required=True, default=True)

    user_name = StringField(required=False)
    tweet_body = StringField(required=False)
    date = DateTimeField(required=False)

    meta = {
        'indexes': [
            'dataset_name',
            'parsed'
        ]
    }