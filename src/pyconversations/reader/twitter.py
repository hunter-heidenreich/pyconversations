import json
from glob import glob

from tqdm import tqdm

from ..convo import Conversation
from ..message import Tweet
from .base import BaseReader


class QuoteReader(BaseReader):

    @staticmethod
    def read(path_pattern, ld=True):
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
        raise NotImplementedError


class ThreadsReader(BaseReader):

    @staticmethod
    def read(path_pattern):
        raise NotImplementedError

    @staticmethod
    def iter_read(path_pattern, ld=True):
        for f in sorted(glob(f'{path_pattern}*tweets.json')):
            convo = Conversation()
            src = f.split('_')[-1].replace('-tweets.json', '')
            tweets = json.load(open(f))
            for tid, tweet in tweets.items():
                xs = Tweet.parse_raw(tweet)
                for x in xs:
                    convo.add_post(x)

            yield src, convo.segment()
