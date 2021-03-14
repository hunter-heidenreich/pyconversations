import json
from glob import glob

from tqdm import tqdm

from ..convo import Conversation
from ..message import FBPost
from .base import BaseReader


class RawFBReader(BaseReader):

    @staticmethod
    def read(path_pattern):
        raise NotImplementedError

    @staticmethod
    def iter_read(path_pattern, ld=True):
        # gather all page names
        pagenames = set()
        for f in glob(path_pattern + '*/'):
            pgname = f.split('/')[-2]
            pagenames.add(pgname)

        for pagename in pagenames:
            page = Conversation()
            for post_path in tqdm(glob(path_pattern + f'{pagename}/*')):
                pid = post_path.split('/')[-1]

                post_del = True
                for f in glob(f'{post_path}/post*.json'):
                    try:
                        post = FBPost.parse_raw(json.load(open(f)), post_type='post', in_reply_to=pagename, lang_detect=ld)

                        if not post:
                            continue

                        pid = post.uid
                        page.add_post(post)
                        post_del = False
                    except json.JSONDecodeError:
                        continue

                # if a post is deleted, let's just create a top-level mock post
                # to keep comments centrally grouped...
                if post_del:
                    post = FBPost(uid=pid, text='[deleted]', author=pagename, platform='Facebook', lang='en')
                    page.add_post(post)

                for f in glob(f'{post_path}/*.json'):
                    if 'post' in f:
                        continue
                    elif 'comments' in f:
                        try:
                            for x in FBPost.parse_raw(json.load(open(f)), post_type='comments', in_reply_to=pid, lang_detect=ld):
                                page.add_post(x)
                        except json.JSONDecodeError:
                            continue
                    elif 'replies' in f:
                        try:
                            for x in FBPost.parse_raw(json.load(open(f)), post_type='replies', in_reply_to=pid, lang_detect=ld):
                                page.add_post(x)
                        except json.JSONDecodeError:
                            # File is corrupt, skip
                            pass
                    elif 'attach' in f:
                        pass
                    elif 'react' in f:
                        pass
                    elif 'scrape' in f:
                        pass
                    else:
                        raise ValueError(f'RawFB::iter_read - Unrecognized file: {f}')

            yield pagename, page.segment()
