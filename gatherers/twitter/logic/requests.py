import logging
from datetime import datetime
from queue import Queue
import aiohttp
import asyncio
import time

from gatherers.twitter.data.database_api import Mongo_API
from gatherers.twitter.logic.parsing import parse_tweet
from gatherers.twitter.utils.async_runner import Async_Runner

# Configure log format and log level
LOG_FORMAT = ('%(levelname) -5s %(asctime) -10s %(name) -5s %(funcName) -5s %(lineno) -10d: %(message)s')
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Twitter request object using a uri and a tweet ID
class Twitter_Request:
    base_uri = 'https://twitter.com/i/web/status/'

    def __init__(self, tweet_id: int):
        self.tweet_id = tweet_id
        self.base_uri = self.base_uri + str(tweet_id)

    # Coroutine function that sends a request for the given tweet and then directly inputs the results into the database
    async def dispatch(self):
        logging.debug("Sending request to url {}.".format(self.base_uri))
        # Aiohttp is used for asynchronous http requests, this results in much greater speeds without resorting to huge
        # thread pools
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_uri) as response:
                response: aiohttp.ClientResponse = response
                text = await response.text()
                logging.info("Received response with status code {}.".format(response.status))

                # If a response code 200 is returned parse the tweet and insert it into the database
                if (response.status == 200):
                    try:
                        parse = parse_tweet(text)
                        date: datetime = datetime.strptime(parse[2], '%I:%M %p - %d %b %Y')
                        logging.debug("Successfully parsed tweet.")
                        Mongo_API().insert_parse_for_tweet(tweet_id=self.tweet_id, tweet_body=parse[0],
                                                           user_name=parse[1], date=date)
                        logging.debug("Successfully inserted tweet contents.")
                    except:
                        Mongo_API().mark_unparseable(self.tweet_id)
                        logging.debug("Failed to parse tweet. Marking tweet as unparseable.")

                # Catch 404 response code "Not found" and mark the tweet as unparsable
                if (response.status == 404):
                    Mongo_API().mark_unparseable(self.tweet_id)
                    logging.debug("Page not found. Marking tweet as unparseable.")

                # Catch 429 response code "Too many requests"
                if (response.status == 429):
                    Mongo_API().mark_unparseable(self.tweet_id)
                    logging.warning("Too many requests sent, consider increasing request sleep time.")

# Dispatcher class that runs in a parallel event loop, dispatches one request from its wait queue every base_wait_time
# seconds
class Twitter_Dispatcher:
    def __init__(self, runner: Async_Runner, dataset: str, base_wait_time: float = 0.002):
        self.pending = Queue()
        self.wait_time = base_wait_time
        self.runner = runner
        self.dataset = dataset
        self.previous_refill = datetime.now()
        self.previous_tweets = Mongo_API().count_total_tweets_processed()
        self.checked_ids = set()
        self.total_time = 0
        self.total_tweets = 0
        runner.add_task(self._wakeup())

    async def _wakeup(self):
        while True:
            # Refill the queue if less than 50 tweets are pending
            if (self.pending.qsize() < 50):
                self._refill()

            # Dispatch a request if there is one pending
            if (self.pending.qsize() > 0):
                request: Twitter_Request = self.pending.get()
                self.runner.add_task(request.dispatch())

            await asyncio.sleep(self.wait_time)

    def _refill(self):
        skip = 0
        ids_added = 0

        # Due to synchronicity concerns, iterate over the parse data until at least 50 items are added to the queue
        while (ids_added < 50):
            self.log_tweets()
            logging.info("Refilling queue.")
            ids = Mongo_API().get_parsable_tweet_ids(dataset_name=self.dataset, skip=skip, limit=250)
            for id in ids:
                if id not in self.checked_ids:
                    self.add_task(Twitter_Request(id))
                    self.checked_ids.add(id)
                    ids_added += 1
                else:
                    skip += 1

    # Log performance status, tweets parsed total and tweets per second for this run
    def log_tweets(self):
        current_refill = datetime.now()
        current_tweets = Mongo_API().count_total_tweets_processed()
        self.total_time += (current_refill - self.previous_refill).total_seconds()
        self.total_tweets += current_tweets - self.previous_tweets

        logging.info("Time since previous refill: {}, Tweets parsed: {}, Tweets per second: {}".format((
            (current_refill - self.previous_refill).total_seconds()),
            current_tweets,
            self.total_tweets / self.total_time))

        self.previous_tweets = current_tweets
        self.previous_refill = current_refill

    # Add a twitter request to the pending queue
    def add_task(self, task: Twitter_Request):
        self.pending.put(task)

    # Get the pending queue size
    def get_pending_tasks(self):
        return self.pending.qsize()


if __name__ == "__main__":
    runner = Async_Runner()
    dispatcher = Twitter_Dispatcher(runner, 'hatespeech-twitter-founta')
