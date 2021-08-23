from .base import BaseTokenizer


class DefaultTokenizer(BaseTokenizer):

    """
    A tokenizer that just uses Python's basic str.split function.
    """

    def __init__(self):
        super(DefaultTokenizer, self).__init__('Default')

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
        return s.split()
