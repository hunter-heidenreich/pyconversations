import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pyconversations.message import ChanPost
from pyconversations.message import FBPost
from pyconversations.message import RedditPost
from pyconversations.message import Tweet
from pyconversations.message.base import get_detector
from pyconversations.reader import ConvoReader
from pyconversations.tokenizers import PartitionTokenizer


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


tokenizers = [PartitionTokenizer]


def char_dist(data):
    df = []
    for size, cnt in tqdm(data['charlen_dist'].items()):
        size = int(size)
        df.extend([{
            'Chars': size
        }] * cnt)
    df = pd.DataFrame(df)
    dsc = df.describe()
    out = 'Char Stats (Latex) & '
    out += display_num(dsc['Chars']['mean']) + ' & '
    out += display_num(dsc['Chars']['std']) + ' & '

    out += display_num(dsc['Chars']['min']) + ' & '
    out += display_num(dsc['Chars']['25%']) + ' & '
    out += display_num(dsc['Chars']['50%']) + ' & '
    out += display_num(dsc['Chars']['75%']) + ' & '
    out += display_num(dsc['Chars']['max']) + ' & '
    out += display_num(data['chars']) + ' \\\\ '
    print(out)

    sns.set_theme()

    sns.histplot(data=df, x='Chars', discrete=True)
    plt.title(f'{title} - Chars per Post', fontsize=18)
    plt.xlabel('Chars', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.savefig(f'out/post_text/img/{tgt}_{args.sel}_posts_char.png', dpi=300)

    plt.clf()

    df['log(Chars)'] = np.log(df['Chars'])
    sns.histplot(data=df, x='log(Chars)')
    plt.title(f'{title} - log(Chars) per Post', fontsize=18)
    plt.xlabel('log(Chars), nats', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.savefig(f'out/post_text/img/{tgt}_{args.sel}_posts_char_log.png', dpi=300)

    plt.clf()


def token_dist(data):
    for tok in tokenizers:
        df = []
        for size, cnt in tqdm(data[tok.NAME]['toklen_dist'].items()):
            size = int(size)
            df.extend([{
                'Tokens': size
            }] * cnt)
        df = pd.DataFrame(df)

        dsc = df.describe()

        out = 'Token Stats (Latex) & '
        out += display_num(dsc['Tokens']['mean']) + ' & '
        out += display_num(dsc['Tokens']['std']) + ' & '

        out += display_num(dsc['Tokens']['min']) + ' & '
        out += display_num(dsc['Tokens']['25%']) + ' & '
        out += display_num(dsc['Tokens']['50%']) + ' & '
        out += display_num(dsc['Tokens']['75%']) + ' & '
        out += display_num(dsc['Tokens']['max']) + ' & '
        out += display_num(sum(data[tok.NAME]['cased'].values())) + ' \\\\ '

        print(out)

        sns.set_theme()

        sns.histplot(data=df, x='Tokens', discrete=True)
        plt.title(f'{title} - Tokens per Post', fontsize=18)
        plt.xlabel('Tokens', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.savefig(f'out/post_text/img/{tgt}_{args.sel}_posts_token.png', dpi=300)

        plt.clf()

        df['log(Tokens)'] = np.log(df['Tokens'])
        sns.histplot(data=df, x='log(Tokens)')
        plt.title(f'{title} - log(Tokens) per Post', fontsize=18)
        plt.xlabel('log(Tokens), nats', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.savefig(f'out/post_text/img/{tgt}_{args.sel}_posts_token_log.png', dpi=300)

        plt.clf()


def type_dist(data):
    for tok in tokenizers:
        df = []
        for size, cnt in tqdm(data[tok.NAME]['typecnt_dist'].items()):
            size = int(size)
            df.extend([{
                'Types': size
            }] * cnt)

        df = pd.DataFrame(df)

        dsc = df.describe()

        out = 'Type Stats (Latex) & '
        out += display_num(dsc['Types']['mean']) + ' & '
        out += display_num(dsc['Types']['std']) + ' & '

        out += display_num(dsc['Types']['min']) + ' & '
        out += display_num(dsc['Types']['25%']) + ' & '
        out += display_num(dsc['Types']['50%']) + ' & '
        out += display_num(dsc['Types']['75%']) + ' & '
        out += display_num(dsc['Types']['max']) + ' & '
        out += display_num(len(data[tok.NAME]['cased'])) + ' \\\\ '

        print(out)

        sns.set_theme()

        sns.histplot(data=df, x='Types', discrete=True)
        plt.title(f'{title} - Types per Post', fontsize=18)
        plt.xlabel('Types', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.savefig(f'out/post_text/img/{tgt}_{args.sel}_posts_type.png', dpi=300)

        plt.clf()

        df['log(Types)'] = np.log(df['Types'])
        sns.histplot(data=df, x='log(Types)')
        plt.title(f'{title} - log(Types) per Post', fontsize=18)
        plt.xlabel('log(Types), nats', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.savefig(f'out/post_text/img/{tgt}_{args.sel}_posts_type_log.png', dpi=300)

        plt.clf()


def type_rank_freq_plot(data, fold='cased'):
    for tok in tokenizers:
        total = sum(data[tok.NAME][fold].values())

        df = []
        for ix, (t, cnt) in enumerate(sorted(data[tok.NAME][fold].items(), key=lambda kv: kv[1], reverse=True)):
            df.append({
                'type': t,
                'rank': ix + 1,
                'log_rank': np.log(ix + 1),
                'freq': cnt / total,
                'log_freq': np.log(cnt) - np.log(total)
            })

        df = pd.DataFrame(df)

        sns.set_theme()
        sns.lineplot(data=df, x='log_rank', y='log_freq')
        plt.title(f'{title} - Type Rank Frequency', fontsize=18)
        plt.xlabel('log(rank)', fontsize=16)
        plt.ylabel('log(freq)', fontsize=16)
        plt.savefig(f'out/post_text/img/{tgt}_{args.sel}_posts_{tok.NAME}_type_logfreq.png', dpi=300)

        plt.clf()


def load(subset='all'):
    try:
        return json.load(open(f'out/post_text/{subset}_{args.sel}_posts_text.json', 'r+'))
    except FileNotFoundError:
        print_every = 100_000

        shell = {'chars': 0, 'charlen_dist': defaultdict(int)}
        for tok in tokenizers:
            shell[tok.NAME] = {
                'cased': {},
                'toklen_dist': defaultdict(int),
                'typecnt_dist': defaultdict(int),
            }

        data = defaultdict(lambda: deepcopy(shell))
        cnt = 0
        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            for post in convo.posts.values():
                if post.lang is None:
                    res = get_detector().FindLanguage(text=post.text)
                    post.lang = res.language if res.is_reliable else 'und'

                lx = len(post.text)
                data[post.lang]['chars'] += lx
                data[post.lang]['charlen_dist'][lx] += 1
                data['all']['chars'] += lx
                data['all']['charlen_dist'][lx] += 1
                if post.lang == 'en' or post.lang == 'und':
                    data['en_und']['chars'] += lx
                    data['en_und']['charlen_dist'][lx] += 1

                for tok in tokenizers:
                    ts = tok.split(post.text)
                    tx = len(ts)
                    tx_ = len(set(ts))
                    data[post.lang][tok.NAME]['toklen_dist'][tx] += 1
                    data[post.lang][tok.NAME]['typecnt_dist'][tx_] += 1
                    data['all'][tok.NAME]['toklen_dist'][tx] += 1
                    data['all'][tok.NAME]['typecnt_dist'][tx_] += 1
                    if post.lang == 'en' or post.lang == 'und':
                        data['en_und'][tok.NAME]['toklen_dist'][tx] += 1
                        data['en_und'][tok.NAME]['typecnt_dist'][tx_] += 1
                    for t in ts:
                        data[post.lang][tok.NAME]['cased'][t] = data[post.lang][tok.NAME]['cased'].get(t, 0) + 1
                        data['all'][tok.NAME]['cased'][t] = data['all'][tok.NAME]['cased'].get(t, 0) + 1
                        if post.lang == 'en' or post.lang == 'und':
                            data['en_und'][tok.NAME]['cased'][t] = data['en_und'][tok.NAME]['cased'].get(t, 0) + 1

                cnt += 1
                if cnt % print_every == 0:
                    print(f'Processed {display_num(cnt)} posts.')

        print('Aggregating token-level stats and distributions')
        for tok in tokenizers:
            # uncased tokens
            print(f'{tok.NAME} -- calculate uncased...')
            for lang in data:
                data[lang][tok.NAME]['uncased'] = defaultdict(int)

                # calculate per lang uncased
                data[lang][tok.NAME]['cased'] = dict(data[lang][tok.NAME]['cased'])

                for term, count in data[lang][tok.NAME]['cased'].items():
                    data[lang][tok.NAME]['uncased'][term.lower()] += count

                data[lang][tok.NAME]['uncased'] = dict(data[lang][tok.NAME]['uncased'])
                data[lang][tok.NAME]['toklen_dist'] = dict(data[lang][tok.NAME]['toklen_dist'])
                data[lang][tok.NAME]['typecnt_dist'] = dict(data[lang][tok.NAME]['typecnt_dist'])
                data[lang]['charlen_dist'] = dict(data[lang]['charlen_dist'])

        print('Writing')
        for lang in tqdm(data):
            json.dump(data[lang], open(f'out/post_text/{lang}_{args.sel}_posts_text.json', 'w+'))

        return data[subset]


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
    os.makedirs('out/post_text/img/', exist_ok=True)

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

    tgt = 'en_und'
    text = load(subset=tgt)
    print('\n' * 5)

    print('-' * 60)
    print(f'Chars: {display_num(text["chars"])}')
    print()
    for tx in tokenizers:
        print(f'{tx.NAME} types (cased): {display_num(len(text[tx.NAME]["cased"]))}')
        print(f'{tx.NAME} tokens (cased): {display_num(sum(text[tx.NAME]["cased"].values()))}')
        print()
        print(f'{tx.NAME} types (uncased): {display_num(len(text[tx.NAME]["uncased"]))}')
        print(f'{tx.NAME} tokens (uncased): {display_num(sum(text[tx.NAME]["uncased"].values()))}')

    char_dist(text)
    token_dist(text)
    type_dist(text)
    type_rank_freq_plot(text, fold='uncased')

    print('-' * 60)
