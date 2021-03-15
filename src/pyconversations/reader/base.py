import json

from abc import ABC
from abc import abstractmethod
from glob import glob

from ..convo import Conversation
from ..message import Tweet


class BaseReader(ABC):

    @staticmethod
    def read(path_pattern):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def iter_read(path_pattern):
        raise NotImplementedError


class ConvoReader(BaseReader):

    @staticmethod
    def read(path_pattern):
        raise NotImplementedError

    @staticmethod
    def iter_read(path_pattern, cons=Tweet):
        for f in glob(path_pattern + '*.json'):
            with open(f) as fp:
                for line in fp.readlines():
                    yield Conversation.from_json(json.loads(line), cons)
