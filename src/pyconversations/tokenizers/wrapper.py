from .base import BaseTokenizer


class LambdaTokenizer(BaseTokenizer):

    """
    An interface that wraps a lambda function
    """

    def __init__(self, func):
        super(LambdaTokenizer, self).__init__('Lambda')
        self._func = func

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
        return self._func(s)
