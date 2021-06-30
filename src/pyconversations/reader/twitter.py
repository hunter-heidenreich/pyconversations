import json
from glob import glob

from tqdm import tqdm

from ..convo import Conversation
from ..message import Tweet
from .base import BaseReader


class QuoteReader(BaseReader):

    """
    A reader specifically designed to read JSONs of Quote tweet archives.
    """

    @staticmethod
    def read(path_pattern, ld=True):
        """
        Reads an entire directory of quote tweet JSONLine files,
        segments them into disjoint conversations,
        and returns the conversations.

        Parameters
        ----------
        path_pattern : str
            The path to the directory
        ld : bool
            Whether to activate language detection (Default: True)

        Returns
        -------
        list(Conversation)
            A list of disjoint conversations
        """
        convo = Conversation()
        for f in sorted(glob(f'{path_pattern}*.json')):
            print(f'Ingesting: {f}')
            with open(f) as fp:
                for line in tqdm(fp.readlines()):
                    for x in Tweet.parse_raw(json.loads(line), lang_detect=ld):
                        convo.add_post(x)
            print(f'In-memory posts: {len(convo.posts)}')

        return convo.segment()

    @staticmethod
    def iter_read(path_pattern, ld=True):
        """
        Function for creating a conversation reading iterator.
        Will read and parse part of a file/directory, yielding segments as queried.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data
        ld : bool
            Whether to activate language detection (Default: True)

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class ThreadsReader(BaseReader):

    """
    This is a custom Twitter "Threads" Reader.
    May be deprecated to adopt new Twitter reply functionality.
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
    def iter_read(path_pattern, ld=True):
        """
        Function for creating a conversation reading iterator.
        Will read and parse part of a file/directory, yielding segments as queried.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data
        ld : bool
            Whether to activate language detection (Default: True)

        Yields
        ------
        2-tuple(str, list(Conversation))
            The string ID of the threaded discussion and a list of the disjoint Conversations identified within it
        """
        for f in sorted(glob(f'{path_pattern}*tweets.json')):
            convo = Conversation()
            src = f.split('_')[-1].replace('-tweets.json', '')
            tweets = json.load(open(f))
            for tid, tweet in tweets.items():
                xs = Tweet.parse_raw(tweet)
                for x in xs:
                    convo.add_post(x)

            yield src, convo.segment()
