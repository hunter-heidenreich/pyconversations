import nltk

from .base import BaseTokenizer


class NLTKTokenizer(BaseTokenizer):
    name = 'NLTK'

    """
    An NLTK-based tokenizer
    """

    @staticmethod
    def tokenize(s):
        """
        Splits a string into tokens.

        Parameters
        ----------
        s : str
            The string to tokenize

        Returns
        -------
        list(str)
            A list of tokens
        """
        return nltk.word_tokenize(s)
