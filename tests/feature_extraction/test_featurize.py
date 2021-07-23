from datetime import datetime

import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction.featurize import ConvFeatures
from pyconversations.feature_extraction.featurize import PostFeatures
from pyconversations.message import Tweet


@pytest.fixture
def mock_post():
    return Tweet(
        uid=13,
        text='@Twitter test this text https://twittr.com',
    )


@pytest.fixture
def mock_convo():
    c = Conversation()

    for ix in range(5):
        c.add_post(Tweet(
            uid=ix, text=f'Text {ix}', reply_to={ix - 1} if ix else None,
            created_at=datetime(year=2020, month=12, day=6, hour=5, minute=1, second=1 + ix)
        ))

    return c


def test_post_get_all_is_dict(mock_post):
    assert type(PostFeatures.get_all(mock_post)) == dict


def test_convo_get_all_is_dict(mock_convo):
    assert type(ConvFeatures.get_all(mock_convo)) == dict


def test_convo_get_all_is_dict_static_and_post(mock_convo):
    fx = ConvFeatures.get_all(mock_convo, include_static=True, include_post_features=True)
    assert type(fx) == dict
