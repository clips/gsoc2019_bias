import logging
from datetime import datetime
from queue import Queue
import aiohttp
import asyncio

from gatherers.twitter.data.database_api import Mongo_API
from gatherers.twitter.logic.parsing import parse_tweet
from gatherers.twitter.utils.async_runner import Async_Runner

LOG_FORMAT = ('%(levelname) -5s %(asctime) -10s %(name) -5s %(funcName) -5s %(lineno) -10d: %(message)s')
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class Twitter_Request:
    base_uri = 'https://twitter.com/i/web/status/'

    def __init__(self, tweet_id : int):
        self.tweet_id = tweet_id
        self.base_uri = self.base_uri + str(tweet_id)

    async def dispatch(self):
        logging.debug("Sending request to url {}.".format(self.base_uri))
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_uri) as response:
                response : aiohttp.ClientResponse = response
                text = await response.text()
                logging.info("Received response with status code {}.".format(response.status))
                if(response.status == 200):
                    try:
                        parse = parse_tweet(text)
                        date : datetime = datetime.strptime(parse[2], '%I:%M %p - %d %b %Y')
                        logging.debug("Successfully parsed tweet.")
                        Mongo_API().insert_parse_for_tweet(tweet_id=self.tweet_id, tweet_body=parse[0], user_name=parse[1], date=date)
                        logging.debug("Successfully inserted tweet contents.")
                    except:
                        Mongo_API().mark_unparseable(self.tweet_id)
                        logging.debug("Failed to parse tweet. Marking tweet as unparseable.")
                if(response.status == 404):
                    Mongo_API().mark_unparseable(self.tweet_id)
                    logging.debug("Page not found. Marking tweet as unparseable.")
                if(response.status == 429):
                    Mongo_API().mark_unparseable(self.tweet_id)
                    logging.warning("Too many requests sent, consider increasing request sleep time.")



class Twitter_Dispatcher:
    def __init__(self, runner : Async_Runner, dataset : str, base_wait_time_millis : float = 0.1):
        self.pending = Queue()
        self.wait_time = base_wait_time_millis
        self.runner = runner
        self.dataset = dataset
        runner.add_task(self._refill())
        runner.add_task(self._wakeup())

    async def _wakeup(self):
        while True:
            if (self.pending.qsize() > 0):
                request : Twitter_Request = self.pending.get()
                self.runner.add_task(request.dispatch())
            await asyncio.sleep(self.wait_time)

    async def _refill(self):
        while True:
            if (self.pending.qsize() < 200):
                logging.info("Refilling queue.")
                for i in range(250):
                    self.add_task(Twitter_Request(Mongo_API().get_parsable_tweet_id(dataset_name=self.dataset, skip=i)))
            await asyncio.sleep(self.wait_time * 50)

    def add_task(self, task: Twitter_Request):
        self.pending.put(task)

    def get_pending_tasks(self):
        return self.pending.qsize()

if __name__ == "__main__":
    runner = Async_Runner()
    dispatcher = Twitter_Dispatcher(runner, 'hatespeech-twitter-founta')