from datetime import datetime as dt

import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.temporal import convo_duration
from pyconversations.feature_extraction.temporal import convo_end_time
from pyconversations.feature_extraction.temporal import convo_start_time
from pyconversations.feature_extraction.temporal import convo_timeseries
from pyconversations.feature_extraction.temporal import post_reply_time
from pyconversations.feature_extraction.temporal import post_to_source
from pyconversations.message import Tweet


@pytest.fixture
def mock_convo():
    c = Conversation()

    for ix in range(5):
        c.add_post(Tweet(uid=ix, text=f'Text {ix}', reply_to={ix - 1} if ix else None,
                         created_at=dt.fromtimestamp(ix)))

    return c


def test_convo_start_time(mock_convo):
    assert convo_start_time(conv=mock_convo, refresh=True) == 0


def test_convo_end_time(mock_convo):
    assert convo_end_time(conv=mock_convo, refresh=True) == 4


def test_convo_duration(mock_convo):
    assert convo_duration(conv=mock_convo, refresh=True) == 4


def test_convo_timeseries(mock_convo):
    assert convo_timeseries(conv=mock_convo, refresh=True) == [0, 1, 2, 3, 4]


def test_reply_time(mock_convo):
    for post in mock_convo.posts.values():
        if post.uid:
            assert post_reply_time(post=post, conv=mock_convo, refresh=True) == 1
        else:
            assert post_reply_time(post=post, conv=mock_convo, refresh=True) == 0


def test_to_source(mock_convo):
    for post in mock_convo.posts.values():
        assert post_to_source(post=post, conv=mock_convo, refresh=True) == post.uid
