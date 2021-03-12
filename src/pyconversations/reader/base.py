from abc import ABC
from abc import abstractmethod


class BaseReader(ABC):

    @staticmethod
    def read(path_pattern):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def iter_read(path_pattern):
        raise NotImplementedError
