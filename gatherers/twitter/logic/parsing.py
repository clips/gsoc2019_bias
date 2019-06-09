from bs4 import BeautifulSoup

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