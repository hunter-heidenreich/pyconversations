import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from glob import glob

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


def load(langs=None):
    try:
        days = {}
        if langs:
            for lang in langs:
                days[lang] = json.load(open(f'out/post_creation/{lang}_{args.ds}_posts_time.json', 'r+'))
        else:
            pths = glob(f'out/post_creation/*_{args.ds}_posts_time.json')

            if not pths:
                raise FileNotFoundError

            for pth in pths:
                lang = pth.split('/')[-1].split('_')[0]
                days[lang] = json.load(open(pth, 'r+'))

        return days
    except FileNotFoundError:
        print('Building...')
        print_every = 250_000
        cnt = 0

        days = defaultdict(list)
        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            for post in convo.posts.values():
                if cnt % print_every == 0:
                    print(f'{display_num(cnt)} posts loaded...')

                cnt += 1

                if not post.created_at:
                    continue

                if post.created_at < min_thresh_dt:
                    continue

                if post.lang is None:
                    res = get_detector().FindLanguage(text=post.text)
                    post.lang = res.language if res.is_reliable else 'und'

                days[post.lang].append(post.created_at.timestamp())
                days['all'].append(post.created_at.timestamp())

        for lang, tx in days.items():
            json.dump(tx, open(f'out/post_creation/{lang}_{args.ds}_posts_time.json', 'w+'))

        if langs:
            days = {lang: tx for lang, tx in days.items() if lang in langs}

        return days


def draw_timestamp(ts, year='all'):
    sns.set_theme()

    size = 8
    aspect = 1.5
    min_dt = ts['Creation Date'].min()
    max_dt = ts['Creation Date'].max()

    g = sns.displot(data=ts, x='Creation Date', height=size, aspect=aspect, kind='kde')

    g.set_xticklabels(rotation=45)
    g.set(xlim=(min_dt, max_dt))
    # y_mx = max([p.get_height() for p in g.ax.patches])
    # g.set(ylim=(0, int(y_mx + 0.1 * y_mx)))

    plt.title(f'{title} - Creation Date Distribution')
    plt.subplots_adjust(bottom=0.1, top=0.95)

    plt.savefig(f'out/post_creation/img/{tgt}_{args.ds}_posts_time_{year}.png')


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
    parser.add_argument('--year', dest='year', type=int, default=None)

    args = parser.parse_args()

    data_root = args.data
    os.makedirs('out/', exist_ok=True)
    min_thresh_dt = datetime(year=2005, month=1, day=1, hour=1, minute=1, second=1)

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

    days = load()
    sel = days['en']

    tgt = 'en_und'
    if tgt == 'en_und':
        sel = days['en'] + days['und']

    df = []
    for ts in sel:
        dx = datetime.fromtimestamp(float(ts))
        df.append({'Creation Date': dx})
    df = pd.DataFrame(df)

    draw_timestamp(df)

    if args.year:
        df = []
        for ts in sel:
            dx = datetime.fromtimestamp(float(ts))

            if dx.year != args.year:
                continue

            df.append({'Creation Date': dx})

        df = pd.DataFrame(df)

        draw_timestamp(df, year=args.year)