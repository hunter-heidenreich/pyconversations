from abc import abstractmethod


class BaseTokenizer:

    """
    The abstract Tokenizer class.
    """

    def __init__(self, name):
        self._name = name

    def __call__(self, s):
        return self.tokenize(s)

    @abstractmethod
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

        Raises
        ------
        NotImplementedError
            Must be implemented in extensions
        """
        raise NotImplementedError
