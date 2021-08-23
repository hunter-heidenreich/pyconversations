from datetime import datetime as dt
from itertools import product

import numpy as np
import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction import ConversationVectorizer
from pyconversations.feature_extraction import PostVectorizer
from pyconversations.feature_extraction import UserVectorizer
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


@pytest.fixture
def all_post_vecs():
    params = product(*[
        [None, 'minmax', 'mean', 'standard'],
    ])

    return [
        PostVectorizer(normalization=n)
        for (n,) in params
    ]


@pytest.fixture
def all_conv_vecs():
    params = product(*[
        [None, 'minmax', 'mean', 'standard'],
    ])

    return [
        ConversationVectorizer(normalization=n)
        for (n,) in params
    ]


@pytest.fixture
def all_user_vecs():
    params = product(*[
        [None, 'minmax', 'mean', 'standard'],
    ])

    return [
        UserVectorizer(normalization=n)
        for (n,) in params
    ]


def test_post_fit_no_args():
    v = PostVectorizer()
    with pytest.raises(ValueError):
        v.fit(None)


def test_post_transform_no_args():
    v = PostVectorizer()
    with pytest.raises(ValueError):
        v.transform(None)


def test_post_vec_with_posts(mock_tweet, all_post_vecs):
    xs = [mock_tweet]
    for v in all_post_vecs:
        v.fit(xs)
        vs = v.transform(xs)

        assert type(vs) == np.ndarray
        assert vs.shape[0] == 1
        assert vs.shape[1] == 18


def test_post_vec_with_convs(mock_convo, all_post_vecs):
    xs = [mock_convo]
    for v in all_post_vecs:

        v.fit(xs)
        vs = v.transform(xs)

        assert type(vs) == np.ndarray
        assert vs.shape[0] == 2
        assert vs.shape[1] == 63


def test_fit_transform_invariance(mock_convo):
    v = PostVectorizer()
    v.fit(mock_convo)
    xs = v.transform(mock_convo)

    v = PostVectorizer()
    xs_ = v.fit_transform(mock_convo)

    assert (xs == xs_).all()


def test_conv_convs_invariance(mock_convo):
    v = PostVectorizer()
    xs = v.fit_transform([mock_convo])

    v = PostVectorizer()
    xs_ = v.fit_transform(mock_convo)

    assert (xs == xs_).all()


def test_post_invalid(mock_convo):
    v = PostVectorizer(normalization='akdfhg;asdhgsd')
    with pytest.raises(ValueError):
        v.fit(mock_convo)

    v = PostVectorizer(normalization='akdfhg;asdhgsd')
    with pytest.raises(KeyError):
        v.transform(mock_convo)


def test_conversation_vec_conv(mock_convo, all_conv_vecs):
    for v in all_conv_vecs:
        xs = v.fit_transform(mock_convo)
        assert type(xs) == np.ndarray


def test_conversation_vec_convs(mock_convo, all_conv_vecs):
    for v in all_conv_vecs:
        xs = v.fit_transform([mock_convo])
        assert type(xs) == np.ndarray


def test_conversation_vec_fail():
    with pytest.raises(ValueError):
        ConversationVectorizer().fit(None)

    with pytest.raises(ValueError):
        ConversationVectorizer().transform(None)


def test_user_vec_conv(mock_convo, all_user_vecs):
    for v in all_user_vecs:
        xs = v.fit_transform(mock_convo)

        assert type(xs) == np.ndarray
        assert xs.shape[0] == 2
        assert xs.shape[1] == 321


def test_user_vec_convs(mock_convo, all_user_vecs):
    for v in all_user_vecs:
        xs = v.fit_transform([mock_convo])

        assert type(xs) == np.ndarray
        assert xs.shape[0] == 2
        assert xs.shape[1] == 320


def test_user_vec_fail():
    with pytest.raises(ValueError):
        UserVectorizer().fit(None)

    with pytest.raises(ValueError):
        UserVectorizer().transform(None)
