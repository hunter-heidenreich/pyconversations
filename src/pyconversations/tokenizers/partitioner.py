import re

from .base import BaseTokenizer

with open('other/chars.txt', 'r') as f:
    CHARS = f.read().strip()
    CHARS = re.sub(' ', '', CHARS)


class PartitionTokenizer(BaseTokenizer):
    NAME = 'Partitioner'
    SPACE = True

    @staticmethod
    def split(s):
        tokens = []
        for token in re.split("([0-9" + CHARS + "'-]+)", s):
            if not PartitionTokenizer.SPACE:
                token = re.sub("[ ]+", "", token)

            if not token:
                continue

            if re.search("[0-9" + CHARS + "'-]", token):
                tokens.append(token)
            else:
                tokens.extend(token)

        return tokens
