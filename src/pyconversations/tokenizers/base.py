from abc import abstractmethod


class BaseTokenizer:
    NAME = 'BaseTokenizer'

    @staticmethod
    @abstractmethod
    def split(s):
        pass
