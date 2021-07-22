import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.dag import convo_connections
from pyconversations.feature_extraction.dag import convo_degrees
from pyconversations.feature_extraction.dag import convo_degrees_dist
from pyconversations.feature_extraction.dag import convo_density
from pyconversations.feature_extraction.dag import convo_depth_dist
from pyconversations.feature_extraction.dag import convo_in_degrees
from pyconversations.feature_extraction.dag import convo_in_degrees_dist
from pyconversations.feature_extraction.dag import convo_internal_nodes
from pyconversations.feature_extraction.dag import convo_leaves
from pyconversations.feature_extraction.dag import convo_messages
from pyconversations.feature_extraction.dag import convo_messages_per_user
from pyconversations.feature_extraction.dag import convo_out_degrees
from pyconversations.feature_extraction.dag import convo_out_degrees_dist
from pyconversations.feature_extraction.dag import convo_post_depth
from pyconversations.feature_extraction.dag import convo_sources
from pyconversations.feature_extraction.dag import convo_tree_degree
from pyconversations.feature_extraction.dag import convo_tree_depth
from pyconversations.feature_extraction.dag import convo_tree_width
from pyconversations.feature_extraction.dag import convo_user_count
from pyconversations.feature_extraction.dag import is_post_internal_node
from pyconversations.feature_extraction.dag import is_post_leaf
from pyconversations.feature_extraction.dag import is_post_source
from pyconversations.feature_extraction.dag import post_degree
from pyconversations.feature_extraction.dag import post_in_degree
from pyconversations.feature_extraction.dag import post_out_degree
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


def test_post_out_degree(mock_convo):
    assert post_out_degree(post=mock_convo.posts[0], conv=mock_convo) == 0


def test_post_out_degree_no_conv(mock_convo):
    assert post_out_degree(post=mock_convo.posts[0]) == 1


def test_post_in_degree(mock_convo):
    assert post_in_degree(post=mock_convo.posts[0], conv=mock_convo) == 1


def test_post_degree(mock_convo):
    assert post_degree(post=mock_convo.posts[0], conv=mock_convo) == 1
    assert post_degree(post=mock_convo.posts[1], conv=mock_convo) == 2


def test_convo_in_degrees(mock_convo):
    assert dict(convo_in_degrees(conv=mock_convo)) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 0}


def test_convo_out_degrees(mock_convo):
    assert dict(convo_out_degrees(conv=mock_convo)) == {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}


def test_convo_degrees(mock_convo):
    assert dict(convo_degrees(conv=mock_convo)) == {0: 1, 1: 2, 2: 2, 3: 2, 4: 1}


def test_convo_in_degrees_dist(mock_convo):
    assert dict(convo_in_degrees_dist(conv=mock_convo)) == {0: 1, 1: 4}


def test_convo_out_degrees_dist(mock_convo):
    assert dict(convo_out_degrees_dist(conv=mock_convo)) == {0: 1, 1: 4}


def test_convo_degrees_dist(mock_convo):
    assert dict(convo_degrees_dist(conv=mock_convo)) == {1: 2, 2: 3}


def test_convo_messages(mock_convo):
    assert convo_messages(conv=mock_convo) == 5


def test_post_is_source(mock_convo):
    assert is_post_source(post=mock_convo.posts[0], conv=mock_convo)
    assert not is_post_source(post=mock_convo.posts[1], conv=mock_convo)


def test_convo_sources(mock_convo):
    assert convo_sources(conv=mock_convo) == 1


def test_post_is_leaf(mock_convo):
    assert not is_post_leaf(post=mock_convo.posts[0], conv=mock_convo)
    assert is_post_leaf(post=mock_convo.posts[4], conv=mock_convo)


def test_convo_leaves(mock_convo):
    assert convo_leaves(conv=mock_convo) == 1


def test_post_is_internal_node(mock_convo):
    assert not is_post_internal_node(post=mock_convo.posts[0], conv=mock_convo)
    assert is_post_internal_node(post=mock_convo.posts[1], conv=mock_convo)


def test_convo_internal_nodes(mock_convo):
    assert convo_internal_nodes(conv=mock_convo) == 3


def test_convo_connections(mock_convo):
    assert convo_connections(conv=mock_convo) == 4


def test_convo_connections_no_check(mock_convo):
    assert convo_connections(conv=mock_convo, check=False) == 4
    assert convo_connections(conv=mock_convo, check=False, refresh=True) == 5


def test_convo_user_count(mock_convo):
    assert convo_user_count(conv=mock_convo) == 2


def test_convo_messages_per_user(mock_convo):
    assert dict(convo_messages_per_user(conv=mock_convo)) == {'USER0': 3, 'USER1': 2}


def test_convo_density(mock_convo):
    assert convo_density(conv=mock_convo) == 0.4


def test_post_depth(mock_convo):
    assert convo_post_depth(post=mock_convo.posts[0], conv=mock_convo) == 0
    assert convo_post_depth(post=mock_convo.posts[1], conv=mock_convo) == 1


def test_convo_depth_dist(mock_convo):
    assert dict(convo_depth_dist(conv=mock_convo)) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}


def test_convo_tree_depth(mock_convo):
    assert convo_tree_depth(conv=mock_convo) == 4


def test_convo_tree_width(mock_convo):
    assert convo_tree_width(conv=mock_convo) == 1


def test_convo_tree_degree(mock_convo):
    assert convo_tree_degree(conv=mock_convo) == 2
