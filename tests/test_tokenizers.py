import pytest

from pyconversations.tokenizers import DefaultTokenizer
from pyconversations.tokenizers import LambdaTokenizer
from pyconversations.tokenizers import NLTKTokenizer
from pyconversations.tokenizers import PartitionTokenizer


@pytest.fixture
def default_tok():
    return DefaultTokenizer()


@pytest.fixture
def nltk_tok():
    return NLTKTokenizer()


@pytest.fixture
def partitioner_tok():
    return PartitionTokenizer()


@pytest.fixture
def no_space_partitioner_tok():
    return PartitionTokenizer(space=False)


@pytest.fixture
def null_charset_partitioner_tok():
    return PartitionTokenizer(charset='')


@pytest.fixture
def lambda_tok():
    return LambdaTokenizer(lambda s: [s])


def abstract_test_tokenizer(tok, tests):
    for i, o in tests:
        assert tok(i) == o


def test_default_tokenizer(default_tok):
    tests = [
        ('test example', ['test', 'example'])
    ]
    abstract_test_tokenizer(default_tok, tests)


def test_nltk(nltk_tok):
    tests = [
        ('test example', ['test', 'example'])
    ]
    abstract_test_tokenizer(nltk_tok, tests)


def test_partitioner(partitioner_tok):
    tests = [
        ('test example', ['test', ' ', 'example'])
    ]
    abstract_test_tokenizer(partitioner_tok, tests)


def test_partitioner_no_space(no_space_partitioner_tok):
    tests = [
        ('test example', ['test', 'example'])
    ]
    abstract_test_tokenizer(no_space_partitioner_tok, tests)


def test_null_charset_partitioner(null_charset_partitioner_tok):
    tests = [
        ('test example', list('test example'))
    ]
    abstract_test_tokenizer(null_charset_partitioner_tok, tests)


def test_lambda(lambda_tok):
    tests = [
        ('test example', ['test example'])
    ]
    abstract_test_tokenizer(lambda_tok, tests)
