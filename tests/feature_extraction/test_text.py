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
    ent = avg_token_entropy_all_splits(post=mock_convo.posts[1], conv=mock_convo)
    assert ent == {
        'after-ancestors':       0.5579833421424286,
        'after-before':          0.5579833421424286,
        'after-children':        0.5644754678724235,
        'after-descendants':     0.5644754678724235,
        'after-full':            0.5579833421424286,
        'after-parents':         0.5579833421424286,
        'after-siblings':        0.5644754678724235,
        'ancestors-after':       0.5391641743406426,
        'ancestors-before':      0.5391641743406426,
        'ancestors-children':    0.5391641743406426,
        'ancestors-descendants': 0.5391641743406426,
        'ancestors-full':        0.5391641743406426,
        'ancestors-parents':     0.5391641743406426,
        'ancestors-siblings':    0.5391641743406426,
        'before-after':          0.6520791811513585,
        'before-ancestors':      0.6520791811513585,
        'before-children':       0.6520791811513585,
        'before-descendants':    0.6520791811513585,
        'before-full':           0.6520791811513585,
        'before-parents':        0.6520791811513585,
        'before-siblings':       0.6520791811513585,
        'children-after':        0.6666666666666666,
        'children-ancestors':    0.6520791811513585,
        'children-before':       0.6520791811513585,
        'children-descendants':  0.6666666666666666,
        'children-full':         0.6520791811513585,
        'children-parents':      0.6520791811513585,
        'children-siblings':     0.6666666666666666,
        'descendants-after':     0.5644754678724235,
        'descendants-ancestors': 0.5579833421424286,
        'descendants-before':    0.5579833421424286,
        'descendants-children':  0.5644754678724235,
        'descendants-full':      0.5579833421424286,
        'descendants-parents':   0.5579833421424286,
        'descendants-siblings':  0.5644754678724235,
        'full-after':            0.5391641743406426,
        'full-ancestors':        0.5391641743406426,
        'full-before':           0.5391641743406426,
        'full-children':         0.5391641743406426,
        'full-descendants':      0.5391641743406426,
        'full-parents':          0.5391641743406426,
        'full-siblings':         0.5391641743406426,
        'parents-after':         0.6520791811513585,
        'parents-ancestors':     0.6520791811513585,
        'parents-before':        0.6520791811513585,
        'parents-children':      0.6520791811513585,
        'parents-descendants':   0.6520791811513585,
        'parents-full':          0.6520791811513585,
        'parents-siblings':      0.6520791811513585,
        'post-after':            0.8710490642551527,
        'post-ancestors':        0.8402708591692185,
        'post-before':           0.9591479170272449,
        'post-children':         0.9591479170272449,
        'post-descendants':      0.8710490642551527,
        'post-full':             0.8402708591692185,
        'post-parents':          0.9591479170272449,
        'post-siblings':         0.8710490642551527,
        'siblings-after':        0.5644754678724235,
        'siblings-ancestors':    0.5579833421424286,
        'siblings-before':       0.5579833421424286,
        'siblings-children':     0.5644754678724235,
        'siblings-descendants':  0.5644754678724235,
        'siblings-full':         0.5579833421424286,
        'siblings-parents':      0.5579833421424286
    }
