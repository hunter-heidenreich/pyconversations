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

    NAME = 'Partitioner'
    SPACE = True
    CHARS = pkgutil.get_data(__package__, 'chars.txt').decode('utf-8').strip().replace(" ", "")

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
        tokens = []
        for token in re.split("([0-9" + PartitionTokenizer.CHARS + "'-]+)", s):
            if not PartitionTokenizer.SPACE:
                token = re.sub("[ ]+", "", token)

            if not token:
                continue

            if re.search("[0-9" + PartitionTokenizer.CHARS + "'-]", token):
                tokens.append(token)
            else:
                tokens.extend(token)

        return tokens
