import pytest

from pyconversations.convo import Conversation
from pyconversations.message import Tweet


@pytest.fixture
def mock_tweet():
    return Tweet(
        uid=1,
        text='test text',
        reply_to={0}
    )


@pytest.fixture
def mock_root_tweet():
    return Tweet(
        uid=0,
        text='root tweet'
    )


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
