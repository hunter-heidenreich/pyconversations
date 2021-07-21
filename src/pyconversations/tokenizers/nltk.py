import nltk

from .base import BaseTokenizer


class NLTKTokenizer(BaseTokenizer):

    """
    An NLTK-based tokenizer
    """

    def __init__(self):
        super(NLTKTokenizer, self).__init__('NLTK')

    def tokenize(self, s):
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
