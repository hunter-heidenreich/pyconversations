import pytest

from datetime import datetime as dt

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.post_in_conv import avg_token_entropy_all_splits
from pyconversations.feature_extraction.post_in_conv import get_all
from pyconversations.feature_extraction.post_in_conv import get_bools
from pyconversations.feature_extraction.post_in_conv import get_floats
from pyconversations.feature_extraction.post_in_conv import get_ints
from pyconversations.feature_extraction.post_in_conv import post_depth
from pyconversations.feature_extraction.post_in_conv import post_reply_time
from pyconversations.feature_extraction.post_in_conv import post_to_source
from pyconversations.feature_extraction.post_in_conv import post_width
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
    for bx in [True, False]:
        for v in get_bools(mock_tweet, mock_convo, include_static=bx).values():
            assert type(v) == bool

        for v in get_floats(mock_tweet, mock_convo, include_static=bx).values():
            assert type(v) == float

        for v in get_ints(mock_tweet, mock_convo, include_static=bx).values():
            assert type(v) == int

        assert type(get_all(mock_tweet, mock_convo, include_static=bx)) == dict


def test_width(mock_tweet, mock_convo):
    assert post_width(mock_tweet, mock_convo) == 1


def test_depth(mock_tweet, mock_convo):
    assert post_depth(mock_tweet, mock_convo) == 1


def test_time(mock_tweet, mock_convo):
    assert post_reply_time(mock_tweet, mock_convo) == 10.0
    assert post_to_source(mock_tweet, mock_convo) == 10.0


def test_entropy_existence(mock_tweet, mock_convo):
    for v in avg_token_entropy_all_splits(mock_tweet, mock_convo).values():
        assert type(v) == float
