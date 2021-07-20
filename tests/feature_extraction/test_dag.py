import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction import DAGFeatures
from pyconversations.message import Tweet


@pytest.fixture
def mock_convo():
    c = Conversation()

    for ix in range(5):
        c.add_post(Tweet(uid=ix, text=f'Text {ix}', reply_to={ix - 1} if ix else {999}))

    return c


@pytest.fixture
def cache():
    return DAGFeatures()


def test_convo_messages(mock_convo, cache):
    assert cache.convo_messages(mock_convo) == 5


def test_convo_connections(mock_convo, cache):
    assert cache.convo_connections(mock_convo) == 4


def test_convo_connections_no_check(mock_convo, cache):
    assert cache.convo_connections(mock_convo, check=False) == 5