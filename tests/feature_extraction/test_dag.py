import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction import DAGFeatures
from pyconversations.message import Tweet


@pytest.fixture
def mock_convo():
    c = Conversation()

    for ix in range(5):
        c.add_post(Tweet(
            uid=ix, text=f'Text {ix}', reply_to={ix - 1} if ix else {999},
            author=f'USER{ix % 2}'
        ))

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


def test_convo_user_count(mock_convo, cache):
    assert cache.convo_user_count(mock_convo) == 2


def test_convo_messages_per_user(mock_convo, cache):
    assert dict(cache.convo_messages_per_user(mock_convo)) == {'USER0': 3, 'USER1': 2}


def test_convo_density(mock_convo, cache):
    assert cache.convo_density(mock_convo) == 0.4


def test_post_out_degree(mock_convo, cache):
    assert cache.post_out_degree(mock_convo.posts[0], conv=mock_convo) == 0


def test_post_out_degree_no_conv(mock_convo, cache):
    assert cache.post_out_degree(mock_convo.posts[0]) == 1


def test_post_in_degree(mock_convo, cache):
    assert cache.post_in_degree(mock_convo.posts[0], mock_convo) == 1


def test_post_depth(mock_convo, cache):
    assert cache.convo_post_depth(mock_convo.posts[0], mock_convo) == 0
    assert cache.convo_post_depth(mock_convo.posts[1], mock_convo) == 1


def test_convo_depth_dist(mock_convo, cache):
    assert dict(cache.convo_depth_dist(mock_convo)) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}


def test_convo_tree_depth(mock_convo, cache):
    assert cache.convo_tree_depth(mock_convo) == 4


def test_convo_tree_width(mock_convo, cache):
    assert cache.convo_tree_width(mock_convo) == 1
