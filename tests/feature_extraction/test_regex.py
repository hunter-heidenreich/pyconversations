import pytest

from pyconversations.feature_extraction import RegexFeatures
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
    return RedditPost(uid=1234, text=txt)


def test_url_patterns(mock_tweet):
    fe = RegexFeatures()
    assert fe.post_url_cnt(mock_tweet) == 1
    assert fe.post_urls(mock_tweet) == ['https://www.twitter.com']


def test_twitter_mention_patterns(mock_tweet):
    fe = RegexFeatures()
    assert fe.post_mention_cnt(mock_tweet) == 1
    assert fe.post_mentions(mock_tweet) == ['@Twitter']


def test_chan_null_mentions():
    null = ChanPost(uid=0)
    fe = RegexFeatures()

    assert fe.post_mention_cnt(null) == 0
    assert fe.post_mentions(null) == []


def test_reddit_mention_patterns(mock_reddit_post):
    fe = RegexFeatures()
    assert fe.post_mention_cnt(mock_reddit_post) == 1
    assert fe.post_mentions(mock_reddit_post) == ['/u/mod']
