import json
from datetime import datetime
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
    def iter_read(path_pattern, ld=True, rd=False):
        convo = Conversation()
        for f in tqdm(sorted(glob(f'{path_pattern}*.json'))):
            if rd:
                with open(f) as fp:
                    for line in fp.readlines():
                        try:
                            data = json.loads(line)
                            convo.add_post(RedditPost.parse_rd(data, lang_detect=ld))
                        except json.decoder.JSONDecodeError:
                            if '}{' in line:
                                lxs = line.split('}{')
                                lx0, lxs = lxs[0], lxs[1:]
                                lx0 += '}'
                                lxs = [lx0] + ['{' + lx for lx in lxs]

                                for lx in lxs:
                                    convo.add_post(RedditPost.parse_rd(json.loads(lx), lang_detect=ld))
                            else:
                                print(line)
                                import pdb
                                pdb.set_trace()

                date_str = f.split('/')[-1][:7]
                dt = datetime.strptime(date_str, '%Y-%m')
                if dt.month in {1, 4, 7, 10}:
                    # dump all  posts older than 6 months
                    out = Conversation()
                    to_drop = set()
                    for uid, post in convo.posts.items():
                        out.add_post(post)
                        to_drop.add(uid)

                    for uid in to_drop:
                        convo.remove_post(uid)

                    out = out.segment()
                    # for o in out:
                    #     o.redact()

                    yield out
            else:
                convo = Conversation()
                with open(f) as fp:
                    for line in fp.readlines():
                        convo.add_post(RedditPost.parse_raw(json.loads(line), lang_detect=ld))

                segs = convo.segment()
                # for s in segs:
                #     s.redact()
                yield segs

        if rd and convo.messages:
            segs = convo.segment()
            # for s in segs:
            #     s.redact()
            yield segs


class BNCReader(BaseReader):

    @staticmethod
    def read(path_pattern, ld=True):
        convo = Conversation()
        for f in tqdm(glob(path_pattern)):
            with open(f) as fp:
                for line in fp.readlines():
                    raw = json.loads(line)
                    post = RedditPost.parse_raw(raw, lang_detect=ld)
                    post.add_tag('AH=1' if raw["violated_rule"] == 2 else 'AH=0')
                    convo.add_post(post)

        segs = convo.segment()
        # for s in segs:
        #     s.redact()
        return segs

    @staticmethod
    def iter_read(path_pattern, ld=True, rd=False):
        raise NotImplementedError
