import numpy as np
import pytest

from pyconversations.convo import Conversation
from pyconversations.message import Tweet


@pytest.fixture
def mock_tweet():
    return Tweet(
        uid=1,
        text='test text',
        author=1,
        reply_to={0}
    )


@pytest.fixture
def mock_root_tweet():
    return Tweet(
        uid=0,
        author=0,
        text='Root tweet text',
    )


@pytest.fixture
def mock_convo(mock_root_tweet):
    convo = Conversation()
    convo.add_post(mock_root_tweet)
    return convo


@pytest.fixture
def mock_convo_path(mock_root_tweet, mock_tweet):
    convo = Conversation()
    convo.add_post(mock_root_tweet)
    convo.add_post(mock_tweet)
    return convo


@pytest.fixture
def mock_multi_convo():
    convo = Conversation()
    for ix in range(10):
        reps = set()
        if ix > 1:
            reps.add(ix - 2)
        convo.add_post(Tweet(uid=ix, reply_to=reps))

    return convo


def test_build_conversation(mock_tweet):
    conversation = Conversation()

    uid = mock_tweet.uid
    rep_to = mock_tweet.reply_to

    conversation.add_post(mock_tweet)
    assert uid in conversation.posts
    assert uid in conversation.edges
    assert conversation.edges[uid] == rep_to

    conversation.remove_post(uid)
    assert uid not in conversation.posts
    assert uid not in conversation.edges

    with pytest.raises(KeyError):
        conversation.remove_post(uid)


def test_add_convo_to_self(mock_tweet):
    conversation = Conversation()

    uid = mock_tweet.uid
    rep_to = mock_tweet.reply_to

    conversation.add_post(mock_tweet)
    conversation = conversation + conversation

    assert uid in conversation.posts
    assert uid in conversation.edges
    assert conversation.edges[uid] == rep_to


def test_add_convo(mock_tweet, mock_root_tweet):
    root_convo = Conversation()
    root_convo.add_post(mock_root_tweet)

    convo = Conversation()
    convo.add_post(mock_tweet)

    full = root_convo + convo

    assert 0 in full.posts
    assert 0 in full.edges
    assert 1 in full.posts
    assert 1 in full.edges
    assert full.edges[0] == set()
    assert full.edges[1] == {0}


def test_convo_segmentation(mock_multi_convo):
    segs = mock_multi_convo.segment()
    assert len(segs) == 2

    even, odd = segs
    for ix in range(5):
        assert 2 * ix in even.posts
        assert (2 * ix) + 1 in odd.posts


def test_to_from_json(mock_convo):
    assert 0 in mock_convo.posts
    assert 0 in mock_convo.edges

    raw_json = mock_convo.to_json()
    new_convo = Conversation()
    new_convo.from_json(raw_json, Tweet)

    assert 0 in new_convo.posts
    assert 0 in new_convo.edges


def test_stats(mock_convo):
    assert mock_convo.messages == 1
    assert mock_convo.messages == 1

    assert mock_convo.connections == 0
    assert mock_convo.connections == 0

    assert mock_convo.users == 1
    assert mock_convo.users == 1

    assert mock_convo.chars == 15
    assert mock_convo.chars == 15

    assert mock_convo.tokens == 3
    assert mock_convo.tokens == 3

    assert mock_convo.token_types == {'Root', 'tweet', 'text'}
    assert mock_convo.token_types == {'Root', 'tweet', 'text'}

    assert mock_convo.token_types_ == {'root', 'tweet', 'text'}
    assert mock_convo.token_types_ == {'root', 'tweet', 'text'}

    assert mock_convo.sources == {0}
    assert mock_convo.sources == {0}

    assert mock_convo.density == 0
    assert mock_convo.degree_hist == [1]
    assert mock_convo.max_clique == 1
    assert mock_convo.avg_cluster == 0
    assert mock_convo.tree_width == 0
    assert mock_convo.assortativity is None


def test_stats_path(mock_convo_path):
    assert mock_convo_path.messages == 2
    assert mock_convo_path.connections == 1
    assert mock_convo_path.users == 2
    assert mock_convo_path.chars == 24
    assert mock_convo_path.tokens == 5
    assert mock_convo_path.token_types_ == {'root', 'tweet', 'text', 'test'}
    assert mock_convo_path.token_types == {'Root', 'tweet', 'text', 'test'}
    assert mock_convo_path.sources == {0}
    assert mock_convo_path.density == 1.0
    assert mock_convo_path.degree_hist == [0, 2]
    assert mock_convo_path.in_degree_hist == {0: 1, 1: 1}
    assert mock_convo_path.out_degree_hist == {0: 1, 1: 1}
    assert mock_convo_path.max_clique == 2
    assert mock_convo_path.avg_cluster == 0
    assert mock_convo_path.tree_width == 1
    assert np.isnan(mock_convo_path.assortativity)

    assert mock_convo_path.centrality_eigen is None
    assert mock_convo_path.centrality_katz == {0: 0.7071067811865476, 1: 0.7071067811865476}
    assert mock_convo_path.centrality_closeness == {0: 1.0, 1: 1.0}
    assert mock_convo_path.centrality_load == {0: 0.0, 1: 0.0}
    assert mock_convo_path.centrality_degree == {0: 1.0, 1: 1.0}
    assert mock_convo_path.centrality_info == {0: 1.0, 1: 1.0}
    assert mock_convo_path.centrality_betweeness == {0: 0.0, 1: 0.0}
    for v in mock_convo_path.centrality_current_flow_betweeness.values():
        assert np.isnan(v)
    assert mock_convo_path.centrality_harmonic == {0: 1.0, 1: 1.0}
    assert mock_convo_path.centrality_second_order == {0: 0.0, 1: 0.0}
    assert mock_convo_path.estrada_index == 3.086161269630487

    assert mock_convo_path.longest_path == 2
    assert mock_convo_path.diameter == 1
    assert mock_convo_path.radius == 1
    assert mock_convo_path.eccentricity == {0: 1, 1: 1}

    assert mock_convo_path.non_randomness == 1.0
    assert np.isnan(mock_convo_path.non_randomness_rel)

    assert mock_convo_path.wiener_index == 1.0
    assert mock_convo_path.closeness_vitality == {0: 1, 1: 1}
    assert mock_convo_path.s_metric == 1.0
    assert mock_convo_path.small_sigma is None
    assert mock_convo_path.small_omega is None

    assert (mock_convo_path.simrank == np.array([[1.0, 0.0], [0.0, 1.0]])).all()

    assert mock_convo_path.rich_club_coefficient is None

    assert mock_convo_path.reciprocity == 0


def test_stats_no_parent(mock_tweet):
    convo = Conversation()
    convo.add_post(mock_tweet)

    assert convo.messages == 1
    assert convo.connections == 0
    assert convo.users == 1
    assert convo.chars == 9
    assert convo.tokens == 2
    assert convo.sources == {1}

    assert convo.density == 0
    assert convo.degree_hist == [1]


def test_conversation_filter_min_char(mock_convo_path):
    assert mock_convo_path.messages == 2
    mock_convo_path.posts[0].text = ''
    mock_convo_path.filter()
    assert mock_convo_path.messages == 1


def test_conversation_filter_by_langs(mock_convo_path):
    assert mock_convo_path.messages == 2
    mock_convo_path.filter(by_langs={'en'})
    assert mock_convo_path.messages == 0


def test_conversation_filter_by_tags(mock_convo_path):
    assert mock_convo_path.messages == 2
    mock_convo_path.filter(by_tags={'#FakeNews'})
    assert mock_convo_path.messages == 0


def test_conversation_filter_by_before(mock_convo_path):
    from datetime import datetime

    assert mock_convo_path.messages == 2
    mock_convo_path.filter(before=datetime(2020, 12, 1, 11, 11, 11))
    assert mock_convo_path.messages == 0


def test_conversation_filter_by_after(mock_convo_path):
    from datetime import datetime

    assert mock_convo_path.messages == 2
    mock_convo_path.filter(after=datetime(2020, 12, 1, 11, 11, 11))
    assert mock_convo_path.messages == 0
