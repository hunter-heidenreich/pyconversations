import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from collections import defaultdict
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

    try:
        days = json.load(open(f'out/{args.ds}_posts_days.json', 'r+'))
    except FileNotFoundError:
        print_every = 250_000

        days = defaultdict(int)
        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            for post in convo.posts.values():
                if not post.created_at:
                    continue

                if post.created_at < min_thresh_dt:
                    continue

                dx = post.created_at
                # days[f'{dx.year}-{dx.month}-{dx.day}'] += 1
                days[dx.timestamp()] += 1

        days = dict(days)
        json.dump(days, open(f'out/{args.ds}_posts_days.json', 'w+'))

    df = []
    for ts, cnt in days.items():
        # y, m, d = [int(x) for x in dstr.split('-')]
        dx = datetime.fromtimestamp(float(ts))
        df.extend([{'Creation Date': dx}] * cnt)
        # df.append({
        #     'Creation Date': dx,
        #     # 'Count': cnt
        # })

    df = pd.DataFrame(df)
    sns.set_theme()

    size = 10
    aspect = 2
    # sns.set(rc={'figure.figsize': (aspect * size, size)})

    # g = sns.barplot(data=df, x='Creation Date', y='Count')  # , height=2, aspect=2)
    g = sns.displot(data=df, x='Creation Date', height=size, aspect=aspect)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_xticklabels(rotation=45)
    plt.title(f'{title} - Creation Date Distribution')
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f'out/{args.ds}_posts_days.png')
