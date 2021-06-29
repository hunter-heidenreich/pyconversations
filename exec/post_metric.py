import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from pyconversations.message import ChanPost
from pyconversations.message import FBPost
from pyconversations.message import RedditPost
from pyconversations.message import Tweet
from pyconversations.message.base import get_detector
from pyconversations.reader import ConvoReader
from pyconversations.tokenizers import PartitionTokenizer
from pyconversations.utils import num2str


def get_post_iterator(print_every=100_000, check_convo=False):
    cnt = 0
    for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
        check = True
        for post in convo.posts.values():
            if cnt and cnt % print_every == 0:
                print(f'Post {num2str(cnt)}')
            cnt += 1

            # double-check lang field
            if post.lang is None or post.lang == 'und':
                res = get_detector().FindLanguage(text=post.text)
                post.lang = res.language if res.is_reliable else 'und'

            if check_convo:
                yield post, convo.messages > 1 and check
                check = False
            else:
                yield post


def load(target='all', metric='lang'):
    if metric == 'lang':
        return load_lang()
    elif metric == 'time':
        return load_time(target)
    elif metric == 'text':
        return load_text(target)
    elif metric == 'size':
        return load_size(target)
    elif metric == 'mu':
        return load_mu(target)
    else:
        raise ValueError(f'post_metric::load - Unrecognized metric `{metric}`')


def load_lang():
    cache_path = f'out/{args.sel}/langs.json'
    try:
        return json.load(open(cache_path))
    except FileNotFoundError:
        lang_lookup = json.load(open('other/langs.json'))

        cnter = defaultdict(int)
        for post in get_post_iterator():
            key = f'{post.lang}:{lang_lookup["main"]["en"]["localeDisplayNames"]["languages"][post.lang]}'
            cnter[key] += 1

        json.dump(dict(cnter), open(cache_path, 'w+'))

        return cnter


def load_time(target):
    cache_path = f'out/{args.sel}/{target}/post/time.json'
    min_thresh_dt = datetime(year=2005, month=1, day=1, hour=1, minute=1, second=1)
    try:
        return json.load(open(cache_path))
    except FileNotFoundError:
        dfs = defaultdict(lambda: defaultdict(int))
        for post in get_post_iterator():
            if not post.created_at:
                continue

            if post.created_at < min_thresh_dt:
                continue

            dfs[post.lang][post.created_at.timestamp()] += 1
            dfs['all'][post.created_at.timestamp()] += 1
            if post.lang == 'en' or post.lang == 'und':
                dfs['en_und'][post.created_at.timestamp()] += 1

        for lang in tqdm(dfs):
            os.makedirs(f'out/{args.sel}/{lang}/post/', exist_ok=True)
            json.dump(dict(dfs[lang]), open(f'out/{args.sel}/{lang}/post/time.json', 'w+'))

        return dfs[target]


def load_text(target):
    try:
        return {
            'freqs': json.load(open(f'out/{args.sel}/{target}/post/freqs.json')),
            'chars': json.load(open(f'out/{args.sel}/{target}/post/chars.json')),
            'tokens': json.load(open(f'out/{args.sel}/{target}/post/tokens.json')),
            'types': json.load(open(f'out/{args.sel}/{target}/post/types.json')),
            'novelty': json.load(open(f'out/{args.sel}/{target}/post/novelty.json')),
        }
    except FileNotFoundError:
        freqs = defaultdict(lambda: defaultdict(int))
        chars = defaultdict(lambda: defaultdict(int))
        tokens = defaultdict(lambda: defaultdict(int))
        types = defaultdict(lambda: defaultdict(int))
        novelty = defaultdict(lambda: defaultdict(int))
        for post in get_post_iterator():
            #  skip blank posts
            if not post.text.strip():
                continue

            langs = [post.lang, 'all']
            if post.lang == 'en' or post.lang == 'und':
                langs.append('en_und')

            ts = post.tokens
            ts_ = post.types
            for lang in langs:
                chars[lang][post.chars] += 1
                tokens[lang][len(ts)] += 1
                types[lang][len(ts_)] += 1
                kx = f'{len(ts)}-{len(ts_)}'
                novelty[lang][kx] += 1

                for t in ts:
                    freqs[lang][t] += 1

        for lang in tqdm(freqs):
            os.makedirs(f'out/{args.sel}/{lang}/post/', exist_ok=True)
            json.dump(dict(freqs[lang]), open(f'out/{args.sel}/{lang}/post/freqs.json', 'w+'))
            json.dump(dict(chars[lang]), open(f'out/{args.sel}/{lang}/post/chars.json', 'w+'))
            json.dump(dict(tokens[lang]), open(f'out/{args.sel}/{lang}/post/tokens.json', 'w+'))
            json.dump(dict(types[lang]), open(f'out/{args.sel}/{lang}/post/types.json', 'w+'))
            json.dump(dict(novelty[lang]), open(f'out/{args.sel}/{lang}/post/novelty.json', 'w+'))

        return {
            'freqs': freqs,
            'chars': chars,
            'tokens': tokens,
            'types': types,
            'novelty': novelty
        }


def load_size(target):
    cache_path = f'out/{args.sel}/{target}/size.json'
    cache_path_temporal = f'out/{args.sel}/{target}/temporal.json'
    min_dt = datetime(year=2005, month=1, day=1, hour=1, minute=1, second=1).timestamp()
    max_dt = datetime(year=2050, month=1, day=1, hour=1, minute=1, second=1).timestamp()
    try:
        return json.load(open(cache_path)), json.load(open(cache_path_temporal))
    except FileNotFoundError:
        size = defaultdict(lambda: defaultdict(int))
        temporal_size = defaultdict(lambda: {
            'start': max_dt,
            'end': min_dt,
            'duration': -1
        })
        for post, is_convo in get_post_iterator(check_convo=True):
            langs = [post.lang, 'all']
            if post.lang == 'en' or post.lang == 'und':
                langs.append('en_und')

            for lang in langs:
                size[lang]['post'] += 1
                size[lang]['convo'] += 1 if is_convo else 0

                if not post.created_at:
                    continue

                upd = False
                if temporal_size[lang]['start'] > post.created_at.timestamp() > min_dt:
                    temporal_size[lang]['start'] = post.created_at.timestamp()
                    upd = True

                if max_dt > post.created_at.timestamp() > temporal_size[lang]['end']:
                    temporal_size[lang]['end'] = post.created_at.timestamp()
                    upd = True

                if upd:
                    temporal_size[lang]['duration'] = temporal_size[lang]['end'] - temporal_size[lang]['start']

        for lang in tqdm(size):
            os.makedirs(f'out/{args.sel}/{lang}/', exist_ok=True)
            json.dump(dict(size[lang]), open(f'out/{args.sel}/{lang}/size.json', 'w+'))
            json.dump(dict(temporal_size[lang]), open(f'out/{args.sel}/{lang}/temporal.json', 'w+'))

        return size[target], temporal_size[target]


def load_mu(target):
    try:
        return json.load(open(f'out/{args.sel}/{target}/post/mu.json'))
    except FileNotFoundError:
        mus = defaultdict(lambda: defaultdict(int))
        for post in get_post_iterator():
            #  skip blank posts
            if not post.text.strip():
                continue

            langs = [post.lang, 'all']
            if post.lang == 'en' or post.lang == 'und':
                langs.append('en_und')

            mu = post.decay_rate
            if mu is None:
                continue

            for lang in langs:
                mus[lang][mu] += 1

        for lang in tqdm(mus):
            os.makedirs(f'out/{args.sel}/{lang}/post/', exist_ok=True)
            json.dump(dict(mus[lang]), open(f'out/{args.sel}/{lang}/post/mu.json', 'w+'))

        return dict(mus)[target]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', dest='data', required=True, type=str, help='General directory data is located in')
    parser.add_argument('--sel', dest='sel', type=str, default='bf',
                        const='bf',
                        nargs='?',
                        choices=[
                            'cmv', 'rd',
                            'ntt', 'ctq',
                            '4chan-news', '4chan-sci', '4chan-his', '4chan-x', '4chan-g', '4chan-pol',
                            'outlets', 'bf',
                            'chan'
                        ],
                        help='Dataset key in selection')
    parser.add_argument('--metric', dest='metric', type=str, default='time',
                        const='time',
                        nargs='?',
                        choices=['time', 'lang', 'text', 'size', 'mu'])

    args = parser.parse_args()

    # other vars
    data_root = args.data
    tokenizers = [PartitionTokenizer]
    tgt_langs = 'en_und'

    # initialize needed structure
    os.makedirs(f'out/{args.sel}', exist_ok=True)

    if args.sel == 'bf':
        dataset = 'FB/BuzzFace/'
        cons = FBPost
        title = 'BuzzFace'
    elif args.sel == 'outlets':
        dataset = 'FB/Outlets/'
        cons = FBPost
        title = 'Outlets'
    elif args.sel == 'chan':
        dataset = '4chan/*/'
        cons = ChanPost
        title = '4Chan'
    elif '4chan' in args.sel:
        dataset = args.sel.replace('-', '/') + '/'
        cons = ChanPost
        title = dataset.replace('4chan', '')
    elif args.sel == 'ctq':
        dataset = 'Twitter/CTQ/'
        cons = Tweet
        title = 'CTQ'
    elif args.sel == 'ntt':
        dataset = 'Twitter/NTT/'
        cons = Tweet
        title = 'NTT'
    elif args.sel == 'cmv':
        dataset = 'Reddit/CMV/'
        cons = RedditPost
        title = 'BNC'
    elif args.sel == 'rd':
        dataset = 'Reddit/RD_*/'
        cons = RedditPost
        title = 'RD'
    else:
        raise ValueError(args)

    data = load(target=tgt_langs, metric=args.metric)
