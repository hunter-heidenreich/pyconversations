import pytest

from pyconversations.feature_extraction import FeatureCache


@pytest.fixture
def fe_cache():
    return FeatureCache()


def test_cache(fe_cache):
    assert fe_cache.get(0, 'test') is None

    fe_cache.cache(0, 'test', -1)
    assert fe_cache.get(0, 'test') == -1

    fe_cache.clear()

    assert fe_cache.get(0, 'test') is None
