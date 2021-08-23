import pkgutil
import re

from .base import BaseTokenizer


class PartitionTokenizer(BaseTokenizer):

    """
    A custom Tokenizer based off of Partitioner by Jake Ryland Williams.

    Notes
    -----
    See for more information: https://github.com/jakerylandwilliams/partitioner
    """

    def __init__(self, space=True, charset=None):
        super(PartitionTokenizer, self).__init__('Partitioner')

        self._space = space
        self._charset = charset

        if self._charset is None:
            self._charset = pkgutil.get_data(__package__, 'chars.txt').decode('utf-8').strip().replace(" ", "")

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
        tokens = []
        for token in re.split("([0-9" + self._charset + "'-]+)", s):
            if not self._space:
                token = re.sub("[ ]+", "", token)

            if not token:
                continue

            if re.search("[0-9" + self._charset + "'-]", token):
                tokens.append(token)
            else:
                tokens.extend(token)

        return tokens
