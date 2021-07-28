import pytest

from datetime import datetime as dt

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.user_in_conv import get_all
from pyconversations.feature_extraction.user_in_conv import get_bools
from pyconversations.feature_extraction.user_in_conv import get_floats
from pyconversations.feature_extraction.user_in_conv import get_ints
from pyconversations.feature_extraction.user_in_conv import mixing_features
from pyconversations.feature_extraction.user_in_conv import novelty_vector
from pyconversations.feature_extraction.user_in_conv import type_frequency_distribution
from pyconversations.message import Tweet


@pytest.fixture
def mock_tweet():
    return Tweet(
        uid=91242213123121,
        text='@Twitter check out this 😏 https://www.twitter.com/ #crazy #link',
        author='apnews',
        reply_to={3894032234},
        created_at=dt(year=2020, month=12, day=12, hour=12, minute=54, second=12)
    )


@pytest.fixture
def mock_convo(mock_tweet):
    cx = Conversation(convo_id='TEST_POST_IN_CONV')
    cx.add_post(mock_tweet)
    cx.add_post(Tweet(
        uid=3894032234,
        text='We are shutting down Twitter',
        author='Twitter',
        created_at=dt(year=2020, month=12, day=12, hour=12, minute=54, second=2)
    ))
    return cx


def test_type_check_extractors(mock_tweet, mock_convo):
    for v in get_bools(mock_tweet.author, mock_convo).values():
        assert type(v) == bool

    for v in get_floats(mock_tweet.author, mock_convo).values():
        assert type(v) == float

    for v in get_ints(mock_tweet.author, mock_convo).values():
        assert type(v) == int

    for bx in [True, False]:
        assert type(get_all(mock_tweet.author, mock_convo, include_post=bx)) == dict


def test_harmonic_feature_existence(mock_tweet, mock_convo):
    freq = type_frequency_distribution(mock_tweet.author, mock_convo)
    novelty = novelty_vector(mock_tweet.author, mock_convo)
    assert len(freq) == len(novelty)

    mix = mixing_features(mock_tweet.author, mock_convo)
    assert type(mix) == dict
    assert len(mix) == 5