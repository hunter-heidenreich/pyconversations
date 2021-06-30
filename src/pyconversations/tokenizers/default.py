from .base import BaseTokenizer


class DefaultTokenizer(BaseTokenizer):
    name = 'Default'

    """
    A tokenizer that just uses Python's basic str.split function.
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
        return s.split()
