import json
from glob import glob

from tqdm import tqdm

from ..convo import Conversation
from ..message import RedditPost
from .base import BaseReader


class RedditReader(BaseReader):

    @staticmethod
    def read(path_pattern):
        raise NotImplementedError

    @staticmethod
    def iter_read(path_pattern, ld=True):
        for f in tqdm(sorted(glob(f'{path_pattern}*.json'))):
            convo = Conversation()
            with open(f) as fp:
                for line in fp.readlines():
                    convo.add_post(RedditPost.parse_raw(json.loads(line), lang_detect=ld))

            segs = convo.segment()
            for s in segs:
                s.redact()
            yield segs
