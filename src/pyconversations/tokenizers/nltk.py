import nltk

from .base import BaseTokenizer


class NLTKTokenizer(BaseTokenizer):
    name = 'NLTK'

    @staticmethod
    def split(s):
        return nltk.word_tokenize(s)
