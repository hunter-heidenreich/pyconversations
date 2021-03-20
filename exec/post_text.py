import json
import os
import re

from argparse import ArgumentParser
from collections import defaultdict
from collections import Counter

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
    parser = ArgumentParser()
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

    tokenizers = [SpaceTokenizer]
    data_root = args.data
    os.makedirs('out/', exist_ok=True)

    if args.ds == 'bf':
        dataset = 'BuzzFace/'
        cons = FBPost
        title = 'BuzzFace'
    elif args.ds == 'outlets':
        dataset = 'Outlets/'
        cons = FBPost
        title = 'Outlets'
    elif '4chan' in args.ds:
        dataset = args.ds.replace('-', '/') + '/'
        cons = ChanPost
        title = dataset.replace('4chan', '')
    elif args.ds == 'ctq':
        dataset = 'CTQuotes/'
        cons = Tweet
        title = 'CTQuotes'
    elif args.ds == 'ntt':
        dataset = 'threads/'
        cons = Tweet
        title = 'NewsTweet'
    elif args.ds == 'cmv':
        dataset = 'Reddit/CMV/'
        cons = RedditPost
        title = 'BNC'
    elif args.ds == 'rd':
        dataset = 'Reddit/RD_*/'
        cons = RedditPost
        title = 'RedditDialog'
    else:
        raise ValueError(args)

    try:
        text = json.load(open(f'out/{args.ds}_posts_text.json', 'r+'))
    except FileNotFoundError:
        print_every = 250_000

        text = {'chars': defaultdict(int)}
        for tok in tokenizers:
            text[tok.NAME] = {
                'cased': defaultdict(Counter),
                'uncased': defaultdict(Counter)
            }

        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            for post in convo.posts.values():
                if post.lang is None:
                    res = get_detector().FindLanguage(text=post.text)
                    post.lang = res.language if res.is_reliable else 'und'

                text['chars'][post.lang] += len(post.text)
                text['chars']['all'] += len(post.text)
                if post.lang == 'en' or post.lang == 'und':
                    text['chars']['en & und'] += len(post.text)

                for tok in tokenizers:
                    ts = Counter(tok.split(post.text))
                    ts_ = Counter(tok.split(post.text.lower()))

                    text[tok.NAME]['cased'][post.lang] += ts
                    text[tok.NAME]['uncased'][post.lang] += ts_

                    text[tok.NAME]['cased']['all'] += ts
                    text[tok.NAME]['uncased']['all'] += ts_

                    if post.lang == 'en' or post.lang == 'und':
                        text[tok.NAME]['cased']['en & und'] += ts
                        text[tok.NAME]['uncased']['en & und'] += ts_

        text['chars'] = dict(text['chars'])
        for tok in tokenizers:
            for lang in text[tok.NAME]['cased']:
                text[tok.NAME]['cased'][lang] = dict(text[tok.NAME]['cased'][lang])
                text[tok.NAME]['uncased'][lang] = dict(text[tok.NAME]['uncased'][lang])
            text[tok.NAME]['cased'] = dict(text[tok.NAME]['cased'])
            text[tok.NAME]['uncased'] = dict(text[tok.NAME]['uncased'])

        json.dump(text, open(f'out/{args.ds}_posts_text.json', 'w+'))

    print('all')
    print(f'Chars: {display_num(text["chars"]["all"])}')
    for tok in tokenizers:
        print(f'{tok.NAME} types (cased): {display_num(len(text[tok.NAME]["cased"]["all"]))}')
        print(f'{tok.NAME} tokens (cased): {display_num(sum(text[tok.NAME]["cased"]["all"].values()))}')

        print(f'{tok.NAME} types (uncased): {display_num(len(text[tok.NAME]["uncased"]["all"]))}')
        print(f'{tok.NAME} tokens (uncased): {display_num(sum(text[tok.NAME]["uncased"]["all"].values()))}')
    print('-' * 60)

    print('en & und')
    print(f'Chars: {display_num(text["chars"]["en & und"])}')
    for tok in tokenizers:
        print(f'{tok.NAME} types (cased): {display_num(len(text[tok.NAME]["cased"]["en & und"]))}')
        print(f'{tok.NAME} tokens (cased): {display_num(sum(text[tok.NAME]["cased"]["en & und"].values()))}')

        print(f'{tok.NAME} types (uncased): {display_num(len(text[tok.NAME]["uncased"]["en & und"]))}')
        print(f'{tok.NAME} tokens (uncased): {display_num(sum(text[tok.NAME]["uncased"]["en & und"].values()))}')
    print('-' * 60)
