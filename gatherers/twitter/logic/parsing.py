from bs4 import BeautifulSoup

# HTML parse for raw tweets using beautiful soup, tries to match the tweet body as well as the username and date with
# HTML filters, this implementation is relatively flimsy, and thus for projects with a greater lifespan the twitter API
# should be used.
def parse_tweet(tweet : str):
    soup = BeautifulSoup(tweet, 'html.parser')
    try:
        tweet_text = soup.find('p', attrs={'class' : 'TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text'}).getText()
    except Exception as exception :
        raise exception

    try:
        user_name = soup.find('strong', attrs={'class' : 'fullname show-popup-with-id u-textTruncate'}).getText()
    except:
        user_name = None

    try:
        date = soup.find('div', attrs={'class' : 'client-and-actions'}).find('span', attrs={'class' : 'metadata'}).find('span').getText()
    except:
        date = None

    return tweet_text, user_name, date