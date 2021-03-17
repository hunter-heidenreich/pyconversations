import re
# import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

from pyconversations.message import *
from pyconversations.message.base import get_detector
from pyconversations.reader import ConvoReader


def display_num(num):
    if num > 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f} T"
    elif num > 1_000_000_000:
        return f"{num / 1_000_000_000:.2f} B"
    elif num > 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif num > 1_000:
        return f"{num / 1_000:.2f} K"
    else:
        if type(num) == int:
            return str(num)

        return str(int(num)) if num.is_integer() else f'{num:.2f}'


class SpaceTokenizer:
    NAME = 'space-separated'

    @staticmethod
    def split(s):
        return re.split(r'\s+', s)


if __name__ == '__main__':
    parser = ArgumentParser('Demo executable of how one might read raw data into conversational format.')
    parser.add_argument('--data', dest='data', required=True, type=str, help='General directory data is located in')
    parser.add_argument('--ds', dest='ds', type=str, default='bf',
                        const='bf',
                        nargs='?',
                        choices=['cmv', 'rd', 'ntt', 'ctq',
                                 '4chan-news', '4chan-sci', '4chan-his', '4chan-x',
                                 '4chan-g', '4chan-pol',
                                 'outlets', 'bf'],
                        help='Dataset key in selection')

    args = parser.parse_args()

    data_root = args.data

    tokenizers = [SpaceTokenizer]

    if args.ds == 'bf':
        dataset = 'BuzzFace/'
        cons = FBPost
    elif args.ds == 'outlets':
        dataset = 'Outlets/'
    elif '4chan' in args.ds:
        dataset = args.ds.replace('-', '/') + '/'
        cons = ChanPost
    elif args.ds == 'ctq':
        dataset = 'CTQuotes/'
        cons = Tweet
    elif args.ds == 'ntt':
        dataset = 'threads/'
        cons = Tweet
    elif args.ds == 'cmv':
        dataset = 'Reddit/CMV/'
        cons = RedditPost
    elif args.ds == 'rd':
        dataset = 'Reddit/RD_*/'
        cons = RedditPost
    else:
        raise ValueError(args)

    print_every = 250_000

    all_posts = {
        'Posts':           0,
        'Conversations':   0,
        'Chars':           0,
        'Tokens':          defaultdict(int),
        'Types (Cased)':   defaultdict(set),
        'Types (Uncased)': defaultdict(set),
        'Start datetime':  datetime(year=2050, month=1, day=1, hour=1, minute=1, second=1),
        'End datetime':    datetime(year=1990, month=1, day=1, hour=1, minute=1, second=1),
    }

    en_posts = deepcopy(all_posts)
    for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
        all_posts['Conversations'] += 1
        en_detect = False

        for post in convo.posts.values():
            if post.lang is None:
                res = get_detector().FindLanguage(text=post.text)
                post.lang = res.language if res.is_reliable else 'und'

            en_bit = post.lang in {'en', 'und'}
            if en_bit and not en_detect:
                en_posts['Conversations'] += 1
                en_detect = True

            all_posts['Posts'] += 1
            if en_bit:
                en_posts['Posts'] += 1
            if all_posts['Posts'] % print_every == 0:
                print(f'Processed {all_posts["Posts"]} posts...')

            all_posts['Chars'] += len(post.text)
            if en_bit:
                en_posts['Chars'] += len(post.text)

            if post.created_at and post.created_at < all_posts['Start datetime']:
                all_posts['Start datetime'] = post.created_at

            if post.created_at and en_bit and post.created_at < en_posts['Start datetime']:
                en_posts['Start datetime'] = post.created_at

            if post.created_at and post.created_at > all_posts['End datetime']:
                all_posts['End datetime'] = post.created_at

            if post.created_at and en_bit and post.created_at > en_posts['End datetime']:
                en_posts['End datetime'] = post.created_at

            for tok in tokenizers:
                ts = tok.split(post.text)
                ts_ = tok.split(post.text.lower())

                all_posts['Tokens'][tok.NAME] += len(ts)
                if en_bit:
                    en_posts['Tokens'][tok.NAME] += len(ts)

                all_posts['Types (Cased)'][tok.NAME] |= set(ts)
                if en_bit:
                    en_posts['Types (Cased)'][tok.NAME] |= set(ts)

                all_posts['Types (Uncased)'][tok.NAME] |= set(ts_)
                if en_bit:
                    en_posts['Types (Uncased)'][tok.NAME] |= set(ts_)

    all_posts['Tokens'] = dict(all_posts['Tokens'])
    all_posts['Types (Cased)'] = {k: len(v) for k, v in all_posts['Types (Cased)'].items()}
    all_posts['Types (Uncased)'] = {k: len(v) for k, v in all_posts['Types (Uncased)'].items()}

    en_posts['Tokens'] = dict(en_posts['Tokens'])
    en_posts['Types (Cased)'] = {k: len(v) for k, v in en_posts['Types (Cased)'].items()}
    en_posts['Types (Uncased)'] = {k: len(v) for k, v in en_posts['Types (Uncased)'].items()}

    t = f"""
            \\hline
            Posts & & {display_num(all_posts["Posts"])} & {display_num(en_posts["Posts"])} ({100 * en_posts["Posts"]/all_posts["Posts"]:.2f}\\%) \\\\
            Conversations & & {display_num(all_posts["Conversations"])} & {display_num(en_posts["Conversations"])} ({100 * en_posts["Conversations"]/all_posts["Conversations"]:.2f}\\%) \\\\
            Chars & & {display_num(all_posts["Chars"])} & {display_num(en_posts["Chars"])} ({100 * en_posts["Chars"]/all_posts["Chars"]:.2f}\\%) \\\\
            \\hline
            Start datetime & & {all_posts["Start datetime"]} & {en_posts["Start datetime"]} \\\\
            End datetime & & {all_posts["End datetime"]} & {en_posts["End datetime"]} \\\\
            \\hline
            Tokens & space-separated & {display_num(all_posts["Tokens"][SpaceTokenizer.NAME])} & {display_num(en_posts["Tokens"][SpaceTokenizer.NAME])} ({100 * en_posts["Tokens"][SpaceTokenizer.NAME]/all_posts["Tokens"][SpaceTokenizer.NAME]:.2f}\\%) \\\\
            Token Types (cased) & space-separated & {display_num(all_posts["Types (Cased)"][SpaceTokenizer.NAME])} & {display_num(en_posts["Types (Cased)"][SpaceTokenizer.NAME])} ({100 * en_posts["Types (Cased)"][SpaceTokenizer.NAME]/all_posts["Types (Cased)"][SpaceTokenizer.NAME]:.2f}\\%) \\\\
            Token Types (uncased) & space-separated & {display_num(all_posts["Types (Uncased)"][SpaceTokenizer.NAME])} & {display_num(en_posts["Types (Uncased)"][SpaceTokenizer.NAME])} ({100 * en_posts["Types (Uncased)"][SpaceTokenizer.NAME]/all_posts["Types (Uncased)"][SpaceTokenizer.NAME]:.2f}\\%) \\\\
            \\hline
    """

    print(t)




