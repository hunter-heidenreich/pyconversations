import json
from datetime import datetime
from glob import glob

from tqdm import tqdm

from ..convo import Conversation
from ..message import RedditPost
from .base import BaseReader


class RedditReader(BaseReader):

    """
    General Reddit raw data reader.
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
    def iter_read(path_pattern, ld=True, rd=False):
        """
        This iterative reading function assumes that the path it will be pointed towards
        contains raw Reddit comments and submissions, sorted/chunked by the month they were created.

        Parameters
        ----------
        path_pattern : str
            The path to the directory containing the data
        ld : bool
            Whether or not activate language detection (Default: True)
        rd : bool
            Whether to use the secondary Reddit parser (`RedditPost.parse_rd`) or not (`RedditPost.parse_raw`) (Default: False)

        Yields
        ------
        list(Conversation)
            A chunk of Conversations, as parsed
        """
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
                    yield out
            else:
                convo = Conversation()
                with open(f) as fp:
                    for line in fp.readlines():
                        convo.add_post(RedditPost.parse_raw(json.loads(line), lang_detect=ld))

                segs = convo.segment()
                yield segs

        if rd and convo.messages:
            segs = convo.segment()
            yield segs


class BNCReader(BaseReader):

    """
    A custom Reddit Reader generated for the data format
    from "Before Name-calling: Dynamics and Triggers of Ad Hominem Fallacies in Web Argumentation" (Habernal et al., 2018).

    Notes
    -----
    See: https://www.aclweb.org/anthology/N18-1036/
    """

    @staticmethod
    def read(path_pattern, ld=True):
        """
        Reads the entire archive of posts from this dataset.
        Posts that violate rule 2 of the r/ChangeMyView sub-reddit are tagged with the `AH=1` tag;
        otherwise, posts are tagged with `AH=0`.

        Parameters
        ----------
        path_pattern : str
            The path to the directory containing the data
        ld : bool
            Whether or not activate language detection (Default: True)

        Returns
        -------
        list(Conversation)
            A list of all parsed and segmented disjoint Conversations within this dataset
        """
        convo = Conversation()
        for f in tqdm(glob(path_pattern)):
            with open(f) as fp:
                for line in fp.readlines():
                    raw = json.loads(line)
                    post = RedditPost.parse_raw(raw, lang_detect=ld)
                    post.add_tag('AH=1' if raw["violated_rule"] == 2 else 'AH=0')
                    convo.add_post(post)

        segs = convo.segment()
        return segs

    @staticmethod
    def iter_read(path_pattern, ld=True, rd=False):
        """
        Function for creating a conversation reading iterator.
        Will read and parse part of a file/directory, yielding segments as queried.

        Parameters
        ----------
        path_pattern : str
            The path to file or directory containing Conversation data
        ld : bool
            Whether or not activate language detection (Default: True)
        rd : bool
            Whether to use the secondary Reddit parser (`RedditPost.parse_rd`) or not (`RedditPost.parse_raw`) (Default: False)

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
