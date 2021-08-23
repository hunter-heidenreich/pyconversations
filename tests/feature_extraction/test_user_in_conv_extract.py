from collections import Counter
from datetime import datetime as dt

import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.user_in_conv import UserInConvoFeatures as UiCF
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
    for v in UiCF.bools(mock_tweet.author, mock_convo).values():
        assert type(v) == bool

    for v in UiCF.floats(mock_tweet.author, mock_convo).values():
        assert type(v) == float

    for v in UiCF.ints(mock_tweet.author, mock_convo).values():
        assert type(v) == int


def test_harmonic_feature_existence(mock_tweet, mock_convo):
    freq = type_frequency_distribution(mock_tweet.author, mock_convo)
    novelty = novelty_vector(mock_tweet.author, mock_convo)
    assert len(freq) == len(novelty)

    mix = mixing_features(mock_tweet.author, mock_convo)
    assert type(mix) == dict
    assert len(mix) == 5


def test_none_user(mock_convo):
    freq = type_frequency_distribution(None, mock_convo)
    assert freq == Counter()
