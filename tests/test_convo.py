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

