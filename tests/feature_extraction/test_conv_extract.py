from datetime import datetime as dt

import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.conv import ConvoFeatures as CF
from pyconversations.feature_extraction.conv import duration
from pyconversations.feature_extraction.conv import mixing_features
from pyconversations.feature_extraction.conv import novelty_vector
from pyconversations.feature_extraction.conv import time_series
from pyconversations.feature_extraction.conv import tree_degree
from pyconversations.feature_extraction.conv import tree_depth
from pyconversations.feature_extraction.conv import tree_width
from pyconversations.feature_extraction.conv import type_frequency_distribution
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


def test_type_check_extractors(mock_convo):
    from collections import Counter
    for v in CF.counter(mock_convo).values():
        assert type(v) == Counter

    for v in CF.floats(mock_convo).values():
        assert type(v) == float

    for v in CF.ints(mock_convo).values():
        assert type(v) == int


def test_harmonic_feature_existence(mock_convo):
    freq = type_frequency_distribution(mock_convo)
    novelty = novelty_vector(mock_convo)
    assert len(freq) == len(novelty)

    mix = mixing_features(mock_convo)
    assert type(mix) == dict
    assert len(mix) == 5


def test_time_series(mock_convo):
    assert time_series(mock_convo, normalize_by_first=True) == [0.0, 10.0]
    assert duration(mock_convo) == time_series(mock_convo, normalize_by_first=True)[-1]
    assert duration(mock_convo) == 10.0


def test_tree_metrics(mock_convo):
    assert tree_degree(mock_convo) == 1
    assert tree_depth(mock_convo) == 1
    assert tree_width(mock_convo) == 1
