import json
import os

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


if __name__ == '__main__':
    parser = ArgumentParser('Demo executable of how one might read raw data into conversational format.')
    parser.add_argument('--data', dest='data', required=True, type=str, help='General directory data is located in')
    parser.add_argument('--ds', dest='ds', type=str, default='bf',
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

    if args.ds == 'bf':
        dataset = 'BuzzFace/'
        cons = FBPost
        title = 'BuzzFace'
    elif args.ds == 'outlets':
        dataset = 'Outlets/'
        cons = FBPost
        title = 'Outlets'
    elif args.ds == 'chan':
        dataset = '4chan/*/'
        cons = ChanPost
        title = '4Chan'
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
        langs = json.load(open(f'out/{args.ds}_posts_langs.json', 'r+'))
    except FileNotFoundError:
        print_every = 250_000

        langs = defaultdict(int)
        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            for post in convo.posts.values():
                if post.lang is None:
                    res = get_detector().FindLanguage(text=post.text)
                    post.lang = res.language if res.is_reliable else 'und'

                langs[post.lang] += 1

        langs = dict(langs)
        json.dump(langs, open(f'out/{args.ds}_posts_langs.json', 'w+'))

    total = sum(langs.values())
    min_thresh = 0.005

    # df = []

    lang_lookup = json.load(open('other/langs.json'))
    o = ''
    for lang, cnt in sorted(langs.items(), key=lambda kv: kv[1], reverse=True):
        if cnt > min_thresh * total:
            # df.append({
            #     'Language':  lang,
            #     'Count': cnt
            # })
            o += f'{lang}:{lang_lookup["main"]["en"]["localeDisplayNames"]["languages"][lang]} ({100 * cnt / total:.2f}\\%), '
    print(o)

    # df = pd.DataFrame(df)
    # sns.set_theme()
    # sns.barplot(data=df, x='Language', y='Count')
    # plt.title(f'{title} - Detected Language')
    # plt.savefig(f'out/{args.ds}_posts_langs.png')
