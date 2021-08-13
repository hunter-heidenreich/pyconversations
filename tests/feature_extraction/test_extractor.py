from datetime import datetime as dt

import numpy as np
import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction import PostVectorizer
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


def test_post_vectorizer_cons():
    x = PostVectorizer()
    assert x._norm is None


def test_fit_no_args():
    v = PostVectorizer()
    with pytest.raises(ValueError):
        v.fit()


def test_transform_no_args():
    v = PostVectorizer()
    with pytest.raises(ValueError):
        v.transform()


def test_post_vec_with_posts(mock_tweet):
    v = PostVectorizer()
    v.fit(posts=[mock_tweet])
    xs = v.transform(posts=[mock_tweet])

    assert type(xs) == np.ndarray


def test_post_vec_with_posts_and_users(mock_tweet):
    v = PostVectorizer(include_user=True)
    v.fit(posts=[mock_tweet])
    xs = v.transform(posts=[mock_tweet])

    assert type(xs) == np.ndarray


def test_post_vec_with_conv(mock_convo):
    v = PostVectorizer(include_conversation=True, include_user=True)
    v.fit(conv=mock_convo)
    xs = v.transform(conv=mock_convo)

    assert type(xs) == np.ndarray


def test_fit_transform_invariance(mock_convo):
    v = PostVectorizer()
    v.fit(conv=mock_convo)
    xs = v.transform(conv=mock_convo)

    v = PostVectorizer()
    xs_ = v.fit_transform(conv=mock_convo)

    assert (xs == xs_).all()


def test_conv_convs_invariance(mock_convo):
    v = PostVectorizer()
    xs = v.fit_transform(convs=[mock_convo])

    v = PostVectorizer()
    xs_ = v.fit_transform(conv=mock_convo)

    assert (xs == xs_).all()


def test_post_minmax(mock_convo):
    v = PostVectorizer(normalization='minmax')
    v.fit(conv=mock_convo)

    kx = 'char_count'
    assert kx in v._stats
    assert 'max' in v._stats[kx]
    assert 'min' in v._stats[kx]
    assert 'range' in v._stats[kx]

    xs = v.transform(conv=mock_convo)
    assert type(xs) == np.ndarray


def test_post_mean(mock_convo):
    v = PostVectorizer(normalization='mean')
    v.fit(conv=mock_convo)

    kx = 'char_count'
    assert kx in v._stats
    assert 'max' in v._stats[kx]
    assert 'min' in v._stats[kx]
    assert 'range' in v._stats[kx]
    assert 'mean' in v._stats[kx]

    xs = v.transform(conv=mock_convo)
    assert type(xs) == np.ndarray


def test_post_standard(mock_convo):
    v = PostVectorizer(normalization='standard')
    v.fit(conv=mock_convo)

    kx = 'char_count'
    assert kx in v._stats
    assert 'std' in v._stats[kx]
    assert 'mean' in v._stats[kx]

    xs = v.transform(conv=mock_convo)
    assert type(xs) == np.ndarray


def test_post_invalid(mock_convo):
    v = PostVectorizer(normalization='akdfhg;asdhgsd')
    with pytest.raises(ValueError):
        v.fit(conv=mock_convo)
    with pytest.raises(ValueError):
        v.transform(conv=mock_convo)
