import pytest

from pyconversations.convo import Conversation
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


@pytest.fixture
def mock_convo():
    c = Conversation()

    for ix in range(5):
        c.add_post(Tweet(uid=ix, text=f'Text {ix}', reply_to={ix - 1} if ix else None))

    return c


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


def test_convo_chars(mock_convo):
    fx = TextFeatures()
    assert fx.convo_chars(mock_convo) == 30
    assert dict(fx.convo_chars_per_post(mock_convo)) == {6: 5}


def test_convo_tokens(mock_convo):
    fx = TextFeatures()

    assert fx.convo_tokens(mock_convo) == 15
    assert dict(fx.convo_tokens_per_post(mock_convo)) == {3: 5}
    assert dict(fx.convo_token_dist(mock_convo)) == {'Text': 5, ' ': 5, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1}


def test_convo_types(mock_convo):
    fx = TextFeatures()
    assert fx.convo_types(mock_convo) == 7
    assert dict(fx.convo_types_per_post(mock_convo)) == {3: 5}
