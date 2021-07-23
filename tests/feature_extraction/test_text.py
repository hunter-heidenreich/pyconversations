from datetime import datetime

import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.text import avg_token_entropy
from pyconversations.feature_extraction.text import avg_token_entropy_all_splits
from pyconversations.feature_extraction.text import convo_chars
from pyconversations.feature_extraction.text import convo_chars_per_post
from pyconversations.feature_extraction.text import convo_token_dist
from pyconversations.feature_extraction.text import convo_tokens
from pyconversations.feature_extraction.text import convo_tokens_per_post
from pyconversations.feature_extraction.text import convo_types
from pyconversations.feature_extraction.text import convo_types_per_post
from pyconversations.feature_extraction.text import post_char_len
from pyconversations.feature_extraction.text import post_tok_dist
from pyconversations.feature_extraction.text import post_tok_len
from pyconversations.feature_extraction.text import post_type_len
from pyconversations.feature_extraction.text import post_types
from pyconversations.message import Tweet


@pytest.fixture
def mock_json_tweet():
    """Returns a mock json of a cached tweet"""
    return {
        'uid':        1234,
        'text':       'This is a tweet! @Twitter',
        'author':     'tweeter1',
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
        c.add_post(Tweet(
            uid=ix, text=f'Text {ix}', reply_to={ix - 1} if ix else None,
            created_at=datetime(year=2020, month=12, day=6, hour=5, minute=1, second=1 + ix)
        ))

    return c


def test_char_len(mock_tweet):
    assert post_char_len(post=mock_tweet) == 25


def test_tok_len(mock_tweet):
    assert post_tok_len(post=mock_tweet) == 10


def test_tok_dist(mock_tweet):
    assert dict(post_tok_dist(post=mock_tweet)) == {' ':        4, 'This': 1, 'is': 1, 'a': 1, 'tweet': 1, '!': 1,
                                                    '@Twitter': 1}


def test_types(mock_tweet):
    assert post_types(post=mock_tweet) == {'!', '@Twitter', 'is', 'a', 'This', ' ', 'tweet'}


def test_type_len(mock_tweet):
    assert post_type_len(post=mock_tweet) == 7


def test_convo_chars(mock_convo):
    assert convo_chars(conv=mock_convo) == 30
    assert dict(convo_chars_per_post(conv=mock_convo)) == {6: 5}


def test_convo_tokens(mock_convo):
    assert convo_tokens(conv=mock_convo) == 15
    assert dict(convo_tokens_per_post(conv=mock_convo)) == {3: 5}
    assert dict(convo_token_dist(conv=mock_convo)) == {'Text': 5, ' ': 5, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1}


def test_convo_types(mock_convo):
    assert convo_types(conv=mock_convo) == 7
    assert dict(convo_types_per_post(conv=mock_convo)) == {3: 5}


def test_avg_token_entropy(mock_convo):
    assert avg_token_entropy(post=mock_convo.posts[0], conv=mock_convo) == 0.8402708591692185


def test_avg_token_entropy_all_splits(mock_convo):
    ent = avg_token_entropy_all_splits(post=mock_convo.posts[1], conv=mock_convo, refresh=True)
    assert type(ent) == dict
