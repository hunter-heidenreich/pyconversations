from datetime import datetime as dt

import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction import TemporalFeatures
from pyconversations.message import Tweet


@pytest.fixture
def mock_convo():
    c = Conversation()

    for ix in range(5):
        c.add_post(Tweet(uid=ix, text=f'Text {ix}', reply_to={ix - 1} if ix else None, created_at=dt.fromtimestamp(ix)))

    return c


@pytest.fixture
def cache():
    return TemporalFeatures()


def test_convo_start_time(mock_convo, cache):
    assert cache.convo_start_time(mock_convo) == 0


def test_convo_end_time(mock_convo, cache):
    assert cache.convo_end_time(mock_convo) == 4


def test_convo_duration(mock_convo, cache):
    assert cache.convo_duration(mock_convo) == 4


def test_convo_timeseries(mock_convo, cache):
    assert cache.convo_timeseries(mock_convo) == [0, 1, 2, 3, 4]


def test_reply_time(mock_convo, cache):
    for post in mock_convo.posts.values():
        if post.uid:
            assert cache.post_reply_time(post, mock_convo) == 1
        else:
            assert cache.post_reply_time(post, mock_convo) == 0


def test_to_source(mock_convo, cache):
    for post in mock_convo.posts.values():
        assert cache.post_to_source(post, mock_convo) == post.uid
