import pytest

from pyconversations.message import UniMessage


def test_from_json_unimessage():
    with pytest.raises(NotImplementedError):
        UniMessage.from_json({})
