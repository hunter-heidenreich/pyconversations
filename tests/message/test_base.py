import pytest

from pyconversations.message import UniMessage


def test_unimplemented_unimessage():
    with pytest.raises(NotImplementedError):
        UniMessage.from_json({})

    with pytest.raises(NotImplementedError):
        UniMessage.parse_raw({})

    with pytest.raises(NotImplementedError):
        UniMessage.parse_datestr('')
