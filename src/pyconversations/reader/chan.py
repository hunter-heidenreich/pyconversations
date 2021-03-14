import json
from glob import glob

from tqdm import tqdm

from ..convo import Conversation
from ..message import ChanPost
from .base import BaseReader


class ChanReader(BaseReader):

    @staticmethod
    def read(path_pattern, ld=True):
        raise NotImplementedError

    @staticmethod
    def iter_read(path_pattern, ld=True):
        for chunk in range(100):
            print(f'Parsing chunk {chunk+1}/100...')

            convo = Conversation()
            for f in glob(path_pattern + f'{chunk:02d}.json'):
                for post in tqdm(json.load(open(f)).values()):
                    px = ChanPost.parse_raw(post, lang_detect=ld)
                    if px:
                        convo.add_post(px)

            yield chunk, convo.segment()
