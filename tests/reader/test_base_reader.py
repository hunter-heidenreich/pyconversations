import pytest

from pyconversations.reader import BaseReader


def test_abstract():
    with pytest.raises(NotImplementedError):
        BaseReader.read('')

    with pytest.raises(NotImplementedError):
        BaseReader.iter_read('')
