import json
from glob import glob

from tqdm import tqdm

from ..convo import Conversation
from ..message import ChanPost
from .base import BaseReader


class ChanReader(BaseReader):

    """
    Reader class for reading and converting raw 4chan data
    """

    @staticmethod
    def read(path_pattern, ld=True):
        """
        Function for reading an entire file/directory of conversations.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data
        ld : bool
            Whether or not language detection should be activated. (Default: True)

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @staticmethod
    def iter_read(path_pattern, ld=True):
        """
        Function for iteratively reading an entire file/directory of conversations.
        Currently expects a `path_pattern` that points to a directory of JSON files
        enumerated from 00 to 99.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data
        ld : bool
            Whether or not language detection should be activated. (Default: True)

        Yields
        ------
        2-tuple(int, Conversation)
            A tuple containing which chunk (in 0..99) this Conversation originated from as well as a Conversation segment.
        """
        for chunk in range(100):
            print(f'Parsing chunk {chunk+1}/100...')

            convo = Conversation()
            for f in glob(path_pattern + f'{chunk:02d}.json'):
                for post in tqdm(json.load(open(f)).values()):
                    px = ChanPost.parse_raw(post, lang_detect=ld)
                    if px:
                        convo.add_post(px)

            yield chunk, convo.segment()
