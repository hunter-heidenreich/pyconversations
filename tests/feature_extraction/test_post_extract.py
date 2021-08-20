import pytest

from pyconversations.feature_extraction.post import PostFeatures as PF
from pyconversations.feature_extraction.post import emojis
from pyconversations.feature_extraction.post import hashtags
from pyconversations.feature_extraction.post import mentions
from pyconversations.feature_extraction.post import mixing_features
from pyconversations.feature_extraction.post import novelty_vector
from pyconversations.feature_extraction.post import out_degree
from pyconversations.feature_extraction.post import type_frequency_distribution
from pyconversations.feature_extraction.post import urls
from pyconversations.message import Tweet


@pytest.fixture
def mock_tweet():
    return Tweet(
        uid=91242213123121,
        text='@Twitter check out this 😏 https://www.twitter.com/ #crazy #link',
        author='apnews',
        reply_to={3894032234},
    )


def test_out_degree(mock_tweet):
    assert out_degree(mock_tweet) == 1


def test_mention_extraction(mock_tweet):
    mxs = mentions(mock_tweet)
    assert len(mxs) == 1
    assert mxs[0] == '@Twitter'


def test_url_extraction(mock_tweet):
    uxs = urls(mock_tweet)
    assert len(uxs) == 1
    assert uxs[0] == 'https://www.twitter.com/'


def test_hashtag_extraction(mock_tweet):
    tags = hashtags(mock_tweet)
    assert len(tags) == 2
    assert '#crazy' in tags
    assert '#link' in tags


def test_emoji_extraction(mock_tweet):
    es = emojis(mock_tweet)
    assert len(es) == 1
    assert es[0] == '😏'


def test_harmonic_feature_existence(mock_tweet):
    freq = type_frequency_distribution(mock_tweet)
    novelty = novelty_vector(mock_tweet)
    assert len(freq) == len(novelty)

    mix = mixing_features(mock_tweet)
    assert type(mix) == dict
    assert len(mix) == 5


def test_type_check_extractors(mock_tweet):
    for v in PF.bools(mock_tweet).values():
        assert type(v) == bool

    for v in PF.categoricals(mock_tweet).values():
        if v is not None:
            assert type(v) == str

    from collections import Counter
    for v in PF.counter(mock_tweet).values():
        assert type(v) == Counter

    for v in PF.floats(mock_tweet).values():
        assert type(v) == float

    for v in PF.ints(mock_tweet).values():
        assert type(v) == int

    for v in PF.strs(mock_tweet).values():
        assert type(v) == list
        for x in v:
            assert type(x) == str
