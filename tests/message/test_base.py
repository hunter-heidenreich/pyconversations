import pytest

from pyconversations.message import UniMessage


def test_unimplemented_unimessage():
    with pytest.raises(NotImplementedError):
        UniMessage.parse_raw({})

    with pytest.raises(NotImplementedError):
        UniMessage.parse_datestr('')


def test_init_tokenizer():
    from pyconversations.message import Tweet

    with pytest.raises(ValueError):
        Tweet(uid=0, tokenizer=None)

    with pytest.raises(KeyError):
        Tweet(uid=0, tokenizer='')

    from pyconversations.tokenizers import LambdaTokenizer

    t = Tweet(uid=0, tokenizer=lambda s: [s])
    assert type(t._tok) == LambdaTokenizer
