import json
import os

import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from pyconversations.message import *
from pyconversations.message.base import get_detector
from pyconversations.reader import ConvoReader
from pyconversations.tokenizers import PartitionTokenizer


def get_post_iterator(print_every=250_000, check_convo=False):
    cnt = 0
    for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
        check = True
        for post in convo.posts.values():
            if cnt and cnt % print_every == 0:
                print(f'Post {cnt}')
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
    else:
        raise ValueError(f'post_metric::load - Unrecognized metric `{metric}`')


def load_lang():
    cache_path = f'out/{args.sel}/langs.csv'
    try:
        return pd.read_csv(cache_path)
    except FileNotFoundError:
        lang_lookup = json.load(open('other/langs.json'))

        df = []
        for post in get_post_iterator():
            df.append({
                'uid': post.uid,
                'lang': post.lang,
                'name': lang_lookup["main"]["en"]["localeDisplayNames"]["languages"][post.lang]
            })

        df = pd.DataFrame(df)
        df.to_csv(cache_path)

        return df


def load_time(target):
    cache_path = f'out/{args.sel}/{target}/post/time.csv'
    min_thresh_dt = datetime(year=2005, month=1, day=1, hour=1, minute=1, second=1)
    try:
        return pd.read_csv(cache_path)
    except FileNotFoundError:
        dfs = defaultdict(list)
        for post in get_post_iterator():
            if not post.created_at:
                continue

            if post.created_at < min_thresh_dt:
                continue

            dfs[post.lang].append({
                'uid': post.uid,
                'creation': post.created_at.timestamp()
            })

        total = None
        en_und = None
        for lang in tqdm(dfs):
            dfs[lang] = pd.DataFrame(dfs[lang])

            if total is None:
                total = pd.DataFrame(dfs[lang])
            else:
                total.append(dfs[lang], ignore_index=True)

            if lang == 'en' or lang == 'und':
                if en_und is None:
                    en_und = pd.DataFrame(dfs[lang])
                else:
                    en_und.append(dfs[lang], ignore_index=True)

            os.makedirs(f'out/{args.sel}/{lang}/post/', exist_ok=True)
            dfs[lang].to_csv(f'out/{args.sel}/{lang}/post/time.csv')

        os.makedirs(f'out/{args.sel}/en_und/post/', exist_ok=True)
        en_und.to_csv(f'out/{args.sel}/en_und/post/time.csv')

        os.makedirs(f'out/{args.sel}/all/post/', exist_ok=True)
        total.to_csv(f'out/{args.sel}/all/post/time.csv')

        if target == 'all':
            return total
        elif target == 'en_und':
            return en_und
        else:
            return dfs[target]


def load_text(target):
    cache_path = f'out/{args.sel}/{target}/post/text.csv'
    try:
        return pd.read_csv(cache_path), json.load(open(f'out/{args.sel}/{target}/post/text_freqs.json'))
    except FileNotFoundError:
        dfs = defaultdict(list)
        freqs = defaultdict(lambda: defaultdict(int))
        for post in get_post_iterator():
            #  skip blank posts
            if not post.text.strip():
                continue
            ts = post.tokens
            ts_ = post.types
            dfs[post.lang].append({
                'uid': post.uid,
                'chars': post.chars,
                'tokens': len(ts),
                'types': len(ts_)
            })

            for t in ts:
                freqs[post.lang][t] += 1
                freqs['all'][t] += 1
                if post.lang == 'en' or post.lang == 'und':
                    freqs['en_und'][t] += 1

        total = None
        en_und = None
        for lang in dfs:
            os.makedirs(f'out/{args.sel}/{lang}/post/', exist_ok=True)

            dfs[lang] = pd.DataFrame(dfs[lang])
            dfs[lang].to_csv(f'out/{args.sel}/{lang}/post/text.csv')

            if total is None:
                total = pd.DataFrame(dfs[lang])
            else:
                total.append(dfs[lang], ignore_index=True)

            if lang == 'en' or lang == 'und':
                if en_und is None:
                    en_und = pd.DataFrame(dfs[lang])
                else:
                    en_und.append(dfs[lang], ignore_index=True)

            freqs[lang] = dict(freqs[lang])
            json.dump(freqs[lang], open(f'out/{args.sel}/{lang}/post/text_freqs.json', 'w+'))

        os.makedirs(f'out/{args.sel}/en_und/post/', exist_ok=True)
        en_und.to_csv(f'out/{args.sel}/en_und/post/text.csv')
        freqs['en_und'] = dict(freqs['en_und'])
        json.dump(freqs['en_und'], open(f'out/{args.sel}/en_und/post/text_freqs.json', 'w+'))

        os.makedirs(f'out/{args.sel}/all/post/', exist_ok=True)
        total.to_csv(f'out/{args.sel}/all/post/text.csv')
        freqs['all'] = dict(freqs['all'])
        json.dump(freqs['all'], open(f'out/{args.sel}/all/post/text_freqs.json', 'w+'))

        if target == 'all':
            return total, freqs[target]
        elif target == 'en_und':
            return en_und, freqs[target]
        else:
            return dfs[target], freqs[target]


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

        for lang in size:
            os.makedirs(f'out/{args.sel}/{lang}/', exist_ok=True)
            json.dump(dict(size[lang]), open(f'out/{args.sel}/{lang}/size.json', 'w+'))
            json.dump(dict(temporal_size[lang]), open(f'out/{args.sel}/{lang}/temporal.json', 'w+'))

        return size[target], temporal_size[target]


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
                        choices=['time', 'lang', 'text', 'size'])

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
