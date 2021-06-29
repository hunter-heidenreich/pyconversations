import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

from pyconversations.message import ChanPost
from pyconversations.message import FBPost
from pyconversations.message import RedditPost
from pyconversations.message import Tweet
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

    args = parser.parse_args()

    data_root = args.data
    os.makedirs('out/', exist_ok=True)

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
        title = 'CTQuotes'
    elif args.sel == 'ntt':
        dataset = 'Twitter/NTT/'
        cons = Tweet
        title = 'NewsTweet'
    elif args.sel == 'cmv':
        dataset = 'Reddit/CMV/'
        cons = RedditPost
        title = 'BNC'
    elif args.sel == 'rd':
        dataset = 'Reddit/RD_*/'
        cons = RedditPost
        title = 'RedditDialog'
    else:
        raise ValueError(args)

    try:
        all_posts = json.load(open(f'out/{args.sel}_size.json', 'r+'))
        all_posts['Start datetime'] = datetime.fromtimestamp(all_posts['Start datetime'])
        all_posts['End datetime'] = datetime.fromtimestamp(all_posts['End datetime'])

        en_posts = json.load(open(f'out/en_und_{args.sel}_size.json', 'r+'))
        en_posts['Start datetime'] = datetime.fromtimestamp(en_posts['Start datetime'])
        en_posts['End datetime'] = datetime.fromtimestamp(en_posts['End datetime'])
    except FileNotFoundError:
        print_every = 250_000

        min_thresh_dt = datetime(year=2005, month=1, day=1, hour=1, minute=1, second=1)
        all_posts = {
            'Posts':           0,
            'Conversations':   0,
            'Start datetime':  datetime(year=2050, month=1, day=1, hour=1, minute=1, second=1),
            'End datetime':    min_thresh_dt,
        }

        en_posts = deepcopy(all_posts)
        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            is_convo = 1 if convo.messages > 1 else 0
            all_posts['Conversations'] += is_convo
            en_detect = False

            for post in convo.posts.values():
                if post.lang is None:
                    res = get_detector().FindLanguage(text=post.text)
                    post.lang = res.language if res.is_reliable else 'und'

                en_bit = post.lang in {'en', 'und'}
                if en_bit and not en_detect:
                    en_posts['Conversations'] += is_convo
                    en_detect = True

                all_posts['Posts'] += 1
                if en_bit:
                    en_posts['Posts'] += 1
                if all_posts['Posts'] % print_every == 0:
                    print(f'Processed {all_posts["Posts"]} posts...')

                if post.created_at and all_posts['Start datetime'] > post.created_at > min_thresh_dt:
                    all_posts['Start datetime'] = post.created_at

                if post.created_at and en_bit and en_posts['Start datetime'] > post.created_at > min_thresh_dt:
                    en_posts['Start datetime'] = post.created_at

                if post.created_at and post.created_at > all_posts['End datetime']:
                    all_posts['End datetime'] = post.created_at

                if post.created_at and en_bit and post.created_at > en_posts['End datetime']:
                    en_posts['End datetime'] = post.created_at

        all_posts['Start datetime'] = all_posts['Start datetime'].timestamp()
        all_posts['End datetime'] = all_posts['End datetime'].timestamp()
        json.dump(all_posts, open(f'out/{args.sel}_size.json', 'w+'))
        all_posts['Start datetime'] = datetime.fromtimestamp(all_posts['Start datetime'])
        all_posts['End datetime'] = datetime.fromtimestamp(all_posts['End datetime'])

        en_posts['Start datetime'] = en_posts['Start datetime'].timestamp()
        en_posts['End datetime'] = en_posts['End datetime'].timestamp()
        json.dump(en_posts, open(f'out/en_und_{args.sel}_size.json', 'w+'))
        en_posts['Start datetime'] = datetime.fromtimestamp(en_posts['Start datetime'])
        en_posts['End datetime'] = datetime.fromtimestamp(en_posts['End datetime'])

    t = f'{args.sel} & '
    t += f'{display_num(all_posts["Posts"])} & {display_num(en_posts["Posts"])} & '
    t += f'{display_num(all_posts["Conversations"])} & {display_num(en_posts["Conversations"])} & '

    s_str = all_posts["Start datetime"].strftime('%Y/%m/%d')
    e_str = all_posts["End datetime"].strftime('%Y/%m/%d')
    delta = all_posts["End datetime"] - all_posts["Start datetime"]
    years = delta.days / 365
    t += f'{s_str} & {e_str} & {years:.2f}  \\\\'

    print(t)
