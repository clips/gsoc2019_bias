import aiohttp
import asyncio

from gatherers.twitter.data.database_api import Mongo_API
from gatherers.twitter.logic.parsing import parse_tweet

class Twitter_Request:
    base_uri = 'https://twitter.com/i/web/status/'

    def __init__(self, tweet_id : int):
        self.tweet_id = tweet_id
        self.base_uri = self.base_uri + str(tweet_id)

    async def dispatch(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_uri) as response:
                text = await response.text()
                try:
                    parse = parse_tweet(text)
                    print(parse[0])
                    print(parse[1])
                    print(parse[2])
                except:
                    pass

class Twitter_Dispatcher:
    pass

request = Twitter_Request(848638037231906817)
asyncio.run(request.dispatch())