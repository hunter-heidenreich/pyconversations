import json
from abc import ABC
from abc import abstractmethod
from glob import glob

from ..convo import Conversation


class BaseReader(ABC):

    """
    Abstract Reader class. Defines the two functions that Readers may implement
    to read from disk.
    """

    @staticmethod
    def read(path_pattern):
        """
        Function for reading an entire file/directory of conversations.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def iter_read(path_pattern):
        """
        Function for creating a conversation reading iterator.
        Will read and parse part of a file/directory, yielding segments as queried.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class ConvoReader(BaseReader):

    """
    Universal Conversation reader.
    Once parsing raw files into the Universal format,
    one can save them to disk and re-load them using this Reader class.
    """

    @staticmethod
    def read(path_pattern):
        """
        Function for reading an entire file/directory of conversations.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @staticmethod
    def iter_read(path_pattern):
        """
        Function for creating a conversation reading iterator.
        Will read and parse part of a file/directory, yielding conversations as queried.

        Parameters
        ----------
        path_pattern : str
            The path to a directory containing Conversation data.
            This path will be appended with the pattern `*.json`.

        Yields
        ------
        Conversation
            A conversation, read from disk.
        """
        for f in glob(path_pattern + '*.json'):
            with open(f) as fp:
                for line in fp.readlines():
                    yield Conversation.from_json(json.loads(line))
