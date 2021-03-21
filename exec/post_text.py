import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

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


def char_dist(subset):
    df = []
    for size, cnt in tqdm(text['charlen_dist'][subset].items()):
        size = int(size)
        df.extend([{
            'Char Len': size
        }] * cnt)
    df = pd.DataFrame(df)
    dsc = df.describe()
    out = display_num(dsc['Char Len']['mean']) + ' & '
    out += display_num(dsc['Char Len']['std']) + ' & '

    out += display_num(dsc['Char Len']['min']) + ' & '
    out += display_num(dsc['Char Len']['25%']) + ' & '
    out += display_num(dsc['Char Len']['50%']) + ' & '
    out += display_num(dsc['Char Len']['75%']) + ' & '
    out += display_num(dsc['Char Len']['max']) + ' \\\\ '

    print(out)

    if filt == 'en':
        sns.set_theme()

        height = 6
        aspect = 1

        mx, mx_ = df['Char Len'].min(), df['Char Len'].max()

        g = sns.displot(data=df, x='Char Len', height=height, aspect=aspect)
        g.set(xlim=(mx, mx_))
        g.ax.set_title(f'{title} - Char Len by Post', fontsize=18)
        g.ax.set_xlabel('# of Chars', fontsize=18)
        g.ax.set_ylabel('# of Posts', fontsize=18)
        g.ax.tick_params(labelsize=12)
        plt.subplots_adjust(top=0.93)

        # plt.show()
        plt.savefig(f'out/{args.ds}_posts_text.png')

        df['log_2(Char Len)'] = np.log2(df['Char Len'])
        mx, mx_ = df['log_2(Char Len)'].min(), df['log_2(Char Len)'].max()

        g = sns.displot(data=df, x='log_2(Char Len)', height=height, aspect=aspect)
        g.set(xlim=(mx, mx_))
        g.ax.set_title(f'{title} - log_2(Char Len) by Post', fontsize=18)
        g.ax.set_xlabel('log_2(# of Chars)', fontsize=18)
        g.ax.set_ylabel('# of Posts', fontsize=18)
        g.ax.tick_params(labelsize=12)
        plt.subplots_adjust(top=0.93)

        # plt.show()
        plt.savefig(f'out/{args.ds}_posts_text_log.png')


def token_dist(subset):
    for tok in tokenizers:
        df = []
        for size, cnt in tqdm(text[tok.NAME]['toklen_dist'][subset].items()):
            size = int(size)
            df.extend([{
                'Token Len': size
            }] * cnt)
        df = pd.DataFrame(df)
        dsc = df.describe()
        out = display_num(dsc['Token Len']['mean']) + ' & '
        out += display_num(dsc['Token Len']['std']) + ' & '

        out += display_num(dsc['Token Len']['min']) + ' & '
        out += display_num(dsc['Token Len']['25%']) + ' & '
        out += display_num(dsc['Token Len']['50%']) + ' & '
        out += display_num(dsc['Token Len']['75%']) + ' & '
        out += display_num(dsc['Token Len']['max']) + ' \\\\ '

        print(out)

        if filt == 'en':
            sns.set_theme()

            height = 6
            aspect = 1

            mx, mx_ = df['Token Len'].min(), df['Token Len'].max()

            g = sns.displot(data=df, x='Token Len', height=height, aspect=aspect)
            g.set(xlim=(mx, mx_))
            g.ax.set_title(f'{title} - Token Len by Post', fontsize=18)
            g.ax.set_xlabel('# of Token', fontsize=18)
            g.ax.set_ylabel('# of Posts', fontsize=18)
            g.ax.tick_params(labelsize=12)
            plt.subplots_adjust(top=0.93)

            # plt.show()
            plt.savefig(f'out/{args.ds}_posts_{tok.NAME}_token.png')

            df['log_2(Token Len)'] = np.log2(df['Token Len'])
            mx, mx_ = df['log_2(Token Len)'].min(), df['log_2(Token Len)'].max()

            g = sns.displot(data=df, x='log_2(Token Len)', height=height, aspect=aspect)
            g.set(xlim=(mx, mx_))
            g.ax.set_title(f'{title} - log_2(Token Len) by Post', fontsize=18)
            g.ax.set_xlabel('log_2(# of Tokens)', fontsize=18)
            g.ax.set_ylabel('# of Posts', fontsize=18)
            g.ax.tick_params(labelsize=12)
            plt.subplots_adjust(top=0.93)

            # plt.show()
            plt.savefig(f'out/{args.ds}_posts_{tok.NAME}_token_log.png')


def type_dist(subset):
    for tok in tokenizers:
        df = []
        for size, cnt in tqdm(text[tok.NAME]['typecnt_dist'][subset].items()):
            size = int(size)
            df.extend([{
                'Type Count': size
            }] * cnt)
        df = pd.DataFrame(df)
        dsc = df.describe()
        out = display_num(dsc['Type Count']['mean']) + ' & '
        out += display_num(dsc['Type Count']['std']) + ' & '

        out += display_num(dsc['Type Count']['min']) + ' & '
        out += display_num(dsc['Type Count']['25%']) + ' & '
        out += display_num(dsc['Type Count']['50%']) + ' & '
        out += display_num(dsc['Type Count']['75%']) + ' & '
        out += display_num(dsc['Type Count']['max']) + ' \\\\ '

        print(out)

        if filt == 'en':
            sns.set_theme()

            height = 6
            aspect = 1

            mx, mx_ = df['Type Count'].min(), df['Type Count'].max()

            g = sns.displot(data=df, x='Type Count', height=height, aspect=aspect)
            g.set(xlim=(mx, mx_))
            g.ax.set_title(f'{title} - Type Count by Post', fontsize=18)
            g.ax.set_xlabel('# of Types', fontsize=18)
            g.ax.set_ylabel('# of Posts', fontsize=18)
            g.ax.tick_params(labelsize=12)
            plt.subplots_adjust(top=0.93)

            # plt.show()
            plt.savefig(f'out/{args.ds}_posts_{tok.NAME}_type.png')

            df['log_2(Type Count)'] = np.log2(df['Type Count'])
            mx, mx_ = df['log_2(Type Count)'].min(), df['log_2(Type Count)'].max()

            g = sns.displot(data=df, x='log_2(Type Count)', height=height, aspect=aspect)
            g.set(xlim=(mx, mx_))
            g.ax.set_title(f'{title} - log_2(Type Count) by Post', fontsize=18)
            g.ax.set_xlabel('log_2(# of Types)', fontsize=18)
            g.ax.set_ylabel('# of Posts', fontsize=18)
            g.ax.tick_params(labelsize=12)
            plt.subplots_adjust(top=0.93)

            # plt.show()
            plt.savefig(f'out/{args.ds}_posts_{tok.NAME}_type_log.png')


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

        text = {'chars': defaultdict(int), 'charlen_dist': defaultdict(lambda: defaultdict(int))}
        for tok in tokenizers:
            text[tok.NAME] = {
                'cased': defaultdict(dict),
                'toklen_dist': defaultdict(lambda: defaultdict(int)),
                'typecnt_dist': defaultdict(lambda: defaultdict(int)),
            }

        cnt = 0
        for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
            for post in convo.posts.values():
                if post.lang is None:
                    res = get_detector().FindLanguage(text=post.text)
                    post.lang = res.language if res.is_reliable else 'und'

                lx = len(post.text)
                text['chars'][post.lang] += lx
                text['charlen_dist'][post.lang][lx] += 1
                for tok in tokenizers:
                    ts = tok.split(post.text)
                    text[tok.NAME]['toklen_dist'][post.lang][len(ts)] += 1
                    text[tok.NAME]['typecnt_dist'][post.lang][len(set(ts))] += 1
                    for t in ts:
                        text[tok.NAME]['cased'][post.lang][t] = text[tok.NAME]['cased'][post.lang].get(t, 0) + 1

                cnt += 1
                if cnt % print_every == 0:
                    print(f'Processed {display_num(cnt)} posts.')

        # aggregate chars
        print('Aggregating character counts')
        text['chars'] = dict(text['chars'])
        text['chars']['all'] = sum(text['chars'].values())
        text['chars']['en & und'] = text['chars'].get('en', 0) + text['chars'].get('und', 0)

        # aggregate char distributions
        print('Unpacking char length distributions')
        text['charlen_dist'] = dict(text['charlen_dist'])
        total = defaultdict(int)
        en_und = defaultdict(int)
        for lang, dist in text['charlen_dist'].items():
            for size, cnt in dist.items():
                total[size] += cnt
                if lang == 'en' or lang == 'und':
                    en_und[size] += cnt
            text['charlen_dist'][lang] = dict(dist)
        text['charlen_dist']['all'] = dict(total)
        text['charlen_dist']['en & und'] = dict(en_und)

        print('Aggregating token-level stats and distributions')
        for tok in tokenizers:
            text[tok.NAME]['uncased'] = {}

            # calculate all tokens
            print(f'{tok.NAME} -- calculate all token cased distribution...')
            text[tok.NAME]['cased']['all'] = {}
            for lang, lang_cnts in text[tok.NAME]['cased'].items():
                for term, cnt in lang_cnts.items():
                    text[tok.NAME]['cased']['all'][term] = text[tok.NAME]['cased']['all'].get(term, 0) + cnt

            # en & und calculation
            text[tok.NAME]['cased']['en & und'] = text[tok.NAME]['cased'].get('en', {})
            for term, cnt in text[tok.NAME]['cased'].get('und', {}).items():
                text[tok.NAME]['cased']['en & und'][term] = text[tok.NAME]['cased']['en & und'].get(term, 0) + cnt

            # uncased tokens
            print(f'{tok.NAME} -- calculate uncased...')
            for lang in text[tok.NAME]['cased']:
                # calculate per lang uncased
                text[tok.NAME]['cased'][lang] = dict(text[tok.NAME]['cased'][lang])
                text[tok.NAME]['uncased'][lang] = defaultdict(int)
                for term, count in text[tok.NAME]['cased'][lang].items():
                    text[tok.NAME]['uncased'][lang][term.lower()] += count
                text[tok.NAME]['uncased'][lang] = dict(text[tok.NAME]['uncased'][lang])

            text[tok.NAME]['cased'] = dict(text[tok.NAME]['cased'])

            # token distribution per post
            print(f'{tok.NAME} -- aggregate token per post dist')
            total = {}
            en_und = {}
            text[tok.NAME]['toklen_dist'] = dict(text[tok.NAME]['toklen_dist'])
            for lang, dist in text[tok.NAME]['toklen_dist'].items():
                for size, cnt in dist.items():
                    total[size] = total.get(size, 0) + cnt
                    if lang == 'en' or lang == 'und':
                        en_und[size] = en_und.get(size, 0) + cnt
                text[tok.NAME]['toklen_dist'][lang] = dict(dist)
            text[tok.NAME]['toklen_dist']['all'] = dict(total)
            text[tok.NAME]['toklen_dist']['en & und'] = dict(en_und)

            # token distribution per post
            print(f'{tok.NAME} -- aggregate type per post dist')
            total = {}
            en_und = {}
            text[tok.NAME]['typecnt_dist'] = dict(text[tok.NAME]['typecnt_dist'])
            for lang, dist in text[tok.NAME]['typecnt_dist'].items():
                for size, cnt in dist.items():
                    total[size] = total.get(size, 0) + cnt
                    if lang == 'en' or lang == 'und':
                        en_und[size] = en_und.get(size, 0) + cnt
                text[tok.NAME]['typecnt_dist'][lang] = dict(dist)
            text[tok.NAME]['typecnt_dist']['all'] = dict(total)
            text[tok.NAME]['typecnt_dist']['en & und'] = dict(en_und)

        json.dump(text, open(f'out/{args.ds}_posts_text.json', 'w+'))
        print('\n' * 5)

    print('-' * 60)
    for filt in ['all', 'en & und', 'en']:
        print(filt)
        print(f'Chars: {display_num(text["chars"][filt])}')
        print()
        for tok in tokenizers:
            print(f'{tok.NAME} types (cased): {display_num(len(text[tok.NAME]["cased"][filt]))}')
            print(f'{tok.NAME} tokens (cased): {display_num(sum(text[tok.NAME]["cased"][filt].values()))}')
            print()
            print(f'{tok.NAME} types (uncased): {display_num(len(text[tok.NAME]["uncased"][filt]))}')
            print(f'{tok.NAME} tokens (uncased): {display_num(sum(text[tok.NAME]["uncased"][filt].values()))}')

        # char_dist(filt)
        # token_dist(filt)
        # type_dist(filt)

        print('-' * 60)
