import pytest

from pyconversations.feature_extraction import TextFeatures
from pyconversations.feature_extraction import post_type_len
from pyconversations.message import Tweet


@pytest.fixture
def mock_json_tweet():
    """Returns a mock json of a cached tweet"""
    return {
        'uid': 1234,
        'text': 'This is a tweet! @Twitter',
        'author': 'tweeter1',
        'created_at': 9999999.0,
        'reply_to':   [1233],
        'platform':   'Twitter',
        'tags':       ['test_tag'],
        'lang':       'en'
    }


@pytest.fixture
def mock_tweet(mock_json_tweet):
    """Returns a mock tweet object"""
    return Tweet(**mock_json_tweet)


def test_char_len(mock_tweet):
    fx = TextFeatures()
    assert fx.post_char_len(mock_tweet) == 25

    # to hit the second wrap condition
    assert fx.post_char_len(mock_tweet) == 25


def test_tok_len(mock_tweet):
    fx = TextFeatures()
    assert fx.post_tok_len(mock_tweet) == 10


def test_tok_dist(mock_tweet):
    fx = TextFeatures()
    assert dict(fx.post_tok_dist(mock_tweet)) == {' ': 4, 'This': 1, 'is': 1, 'a': 1, 'tweet': 1, '!': 1, '@Twitter': 1}


def test_types(mock_tweet):
    fx = TextFeatures()
    assert fx.post_types(mock_tweet) == {'!', '@Twitter', 'is', 'a', 'This', ' ', 'tweet'}
    assert fx.post_types(mock_tweet) == {'!', '@Twitter', 'is', 'a', 'This', ' ', 'tweet'}

    fx.clear()
    fx.post_tok_dist(mock_tweet)
    assert fx.post_types(mock_tweet) == {'!', '@Twitter', 'is', 'a', 'This', ' ', 'tweet'}


def test_type_len(mock_tweet):
    fx = TextFeatures()
    assert fx.post_type_len(mock_tweet) == 7
    assert fx.post_type_len(mock_tweet) == 7
    assert post_type_len(mock_tweet) == 7
