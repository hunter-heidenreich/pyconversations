from datetime import datetime

import pytest

from pyconversations.convo import Conversation
from pyconversations.feature_extraction import ConversationFeaturizer
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


@pytest.fixture
def conv_ext():
    return ConversationFeaturizer(include_post=True)


def test_convo_get_all_is_dict_static_and_post(mock_convo, conv_ext):
    fx = conv_ext.transform(mock_convo)
    assert type(fx) == dict
