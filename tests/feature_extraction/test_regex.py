import pytest

from pyconversations.feature_extraction.regex import post_urls
from pyconversations.feature_extraction.regex import post_user_mentions
from pyconversations.message import ChanPost
from pyconversations.message import RedditPost
from pyconversations.message import Tweet


@pytest.fixture
def mock_tweet():
    """Returns a mock tweet object"""
    text = '@Twitter pls check this link out https://www.twitter.com'
    return Tweet(uid=1234, text=text)


@pytest.fixture
def mock_reddit_post():
    txt = '/u/mod get outta here'
    return RedditPost(uid=12349999, text=txt)


def test_url_patterns(mock_tweet):
    cnt, urls = post_urls(post=mock_tweet)
    assert cnt == 1
    assert urls == ['https://www.twitter.com']


def test_twitter_mention_patterns(mock_tweet):
    cnt, mentions = post_user_mentions(post=mock_tweet)
    assert cnt == 1
    assert mentions == ['@Twitter']


def test_chan_null_mentions():
    null = ChanPost(uid=0)
    cnt, mentions = post_user_mentions(post=null)

    assert cnt == 0
    assert mentions == []


def test_reddit_mention_patterns(mock_reddit_post):
    cnt, mentions = post_user_mentions(post=mock_reddit_post)
    assert cnt == 1
    assert mentions == ['/u/mod']
