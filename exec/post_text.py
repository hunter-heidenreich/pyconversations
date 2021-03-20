import json
import os
import re

from argparse import ArgumentParser
from collections import defaultdict

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
        print_every = 100_000

        text = {'chars': defaultdict(int)}
        for tok in tokenizers:
            text[tok.NAME] = {
                'cased': defaultdict(dict)
            }

        cnt = 0
        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            for post in convo.posts.values():
                if post.lang is None:
                    res = get_detector().FindLanguage(text=post.text)
                    post.lang = res.language if res.is_reliable else 'und'

                text['chars'][post.lang] += len(post.text)
                for tok in tokenizers:
                    for t in tok.split(post.text):
                        text[tok.NAME]['cased'][post.lang][t] = text[tok.NAME]['cased'][post.lang].get(t, 0) + 1

                cnt += 1
                if cnt % print_every == 0:
                    print(f'Processed {display_num(cnt)} posts.')

        # aggregate chars
        text['chars'] = dict(text['chars'])
        text['chars']['all'] = sum(text['chars'].values())
        text['chars']['en & und'] = text['chars'].get('en', 0) + text['chars'].get('und', 0)

        for tok in tokenizers:
            text[tok.NAME]['uncased'] = {}

            # calculate all tokens
            print(f'{tok.NAME} -- calculate all token cased distribution...')
            text[tok.NAME]['cased']['all'] = {}
            for lang_cnts in text[tok.NAME]['cased'].values():
                for term, cnt in lang_cnts.items():
                    text[tok.NAME]['cased']['all'][term] = text[tok.NAME]['cased']['all'].get(term, 0) + cnt

            # en & und calculation
            text[tok.NAME]['cased']['en & und'] = text[tok.NAME]['cased'].get('en', {})
            for term, cnt in text[tok.NAME]['cased'].get('und', {}).items():
                text[tok.NAME]['cased']['en & und'][term] = text[tok.NAME]['cased']['en & und'].get(term, 0) + cnt

            print(f'{tok.NAME} -- calculate uncased...')
            for lang in text[tok.NAME]['cased']:
                print(f'\tAggregating {lang}')

                # calculate per lang uncased
                text[tok.NAME]['cased'][lang] = dict(text[tok.NAME]['cased'][lang])
                text[tok.NAME]['uncased'][lang] = defaultdict(int)
                for term, count in text[tok.NAME]['cased'][lang].items():
                    t_ = term.lower()
                    text[tok.NAME]['uncased'][lang][t_] += count
                text[tok.NAME]['uncased'][lang] = dict(text[tok.NAME]['uncased'][lang])

            text[tok.NAME]['cased'] = dict(text[tok.NAME]['cased'])

        json.dump(text, open(f'out/{args.ds}_posts_text.json', 'w+'))
        print('\n' * 5)

    print('-' * 60)
    for filt in ['all', 'en & und', 'en']:
        print(filt)
        print(f'Chars: {display_num(text["chars"][filt])}')
        for tok in tokenizers:
            print(f'{tok.NAME} types (cased): {display_num(len(text[tok.NAME]["cased"][filt]))}')
            print(f'{tok.NAME} tokens (cased): {display_num(sum(text[tok.NAME]["cased"][filt].values()))}')

            print(f'{tok.NAME} types (uncased): {display_num(len(text[tok.NAME]["uncased"][filt]))}')
            print(f'{tok.NAME} tokens (uncased): {display_num(sum(text[tok.NAME]["uncased"][filt].values()))}')
        print('-' * 60)
