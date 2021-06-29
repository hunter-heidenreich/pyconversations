import json
import os
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm

from pyconversations.message import ChanPost
from pyconversations.message import FBPost
from pyconversations.message import RedditPost
from pyconversations.message import Tweet
from pyconversations.message.base import get_detector
from pyconversations.reader import ConvoReader
from pyconversations.utils import num2str


def get_convo_iterator(print_every=100_000):
    cnt = 0
    for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
        langs = set()
        for post in convo.posts.values():
            if cnt and cnt % print_every == 0:
                print(f'Parsing post {num2str(cnt)}')
            cnt += 1

            # double-check lang field
            if post.lang is None or post.lang == 'und':
                res = get_detector().FindLanguage(text=post.text)
                post.lang = res.language if res.is_reliable else 'und'
            langs.add(post.lang)

        yield convo, langs


def load(target='all', metric='size'):
    if metric == 'size':
        return load_size(target)
    elif metric == 'text':
        return load_text(target)
    elif metric == 'graph':
        return load_graph(target)
    elif metric == 'mu':
        return load_mu(target)
    else:
        raise ValueError(f'post_metric::load - Unrecognized metric `{metric}`')


def load_size(target):
    try:
        cache_path = f'out/{args.sel}/{target}/convo/size.json'
        return json.load(open(cache_path))
    except FileNotFoundError:
        keys = ['messages', 'langs', 'users', 'duration']
        x = {k: defaultdict(lambda: defaultdict(int)) for k in keys}
        for convo, langs in get_convo_iterator():
            if convo.messages < 2:
                continue

            langs.add('all')
            extra = 1
            if 'en' in langs or 'und' in langs:
                langs.add('en_und')
                extra += 1

            for lang in langs:
                x['messages'][lang][convo.messages] += 1
                x['langs'][lang][len(langs) - extra] += 1
                x['users'][lang][convo.users] += 1
                x['duration'][lang][convo.duration] += 1

        for lang in x['langs']:
            out = {k: dict(x[k][lang]) for k in keys}
            os.makedirs(f'out/{args.sel}/{lang}/convo/', exist_ok=True)
            json.dump(out, open(f'out/{args.sel}/{lang}/convo/size.json', 'w+'))

        return {k: dict(x[k][target]) for k in keys}


def load_text(target):
    try:
        cache_path = f'out/{args.sel}/{target}/convo/text.json'
        return json.load(open(cache_path))
    except FileNotFoundError:
        keys = ['chars', 'tokens', 'types', 'novelty']
        x = {k: defaultdict(lambda: defaultdict(int)) for k in keys}
        for convo, langs in get_convo_iterator():
            if convo.messages < 2:
                continue

            langs.add('all')
            extra = 1
            if 'en' in langs or 'und' in langs:
                langs.add('en_und')
                extra += 1

            chars = convo.chars
            tokens = convo.tokens
            types = convo.token_types
            kx = f'{tokens}-{types}'

            for lang in langs:
                x['chars'][lang][chars] += 1
                x['tokens'][lang][tokens] += 1
                x['types'][lang][types] += 1
                x['novelty'][lang][kx] += 1

        for lang in x['chars']:
            out = {k: dict(x[k][lang]) for k in keys}
            os.makedirs(f'out/{args.sel}/{lang}/convo/', exist_ok=True)
            json.dump(out, open(f'out/{args.sel}/{lang}/convo/text.json', 'w+'))

        return {k: dict(x[k][target]) for k in keys}


def load_graph(target):
    try:
        cache_path = f'out/{args.sel}/{target}/convo/graph.json'
        return json.load(open(cache_path))
    except FileNotFoundError:
        keys = [
            'tree_depth', 'tree_width',
            'degrees'
        ]
        x = {k: defaultdict(lambda: defaultdict(int)) for k in keys}
        for convo, langs in get_convo_iterator():
            if convo.messages < 2:
                continue

            langs.add('all')
            extra = 1
            if 'en' in langs or 'und' in langs:
                langs.add('en_und')
                extra += 1

            rep_degree_strings = [f'{total}-{rin}-{rout}' for (total, rin, rout) in convo.reply_counts]

            for lang in langs:
                # x['density'][lang][convo.density] += 1

                # x['avg_degree'][lang][np.average(convo.degree_hist)] += 1
                # x['avg_in_degree'][lang][np.average(convo.in_degree_hist)] += 1
                # x['avg_out_degree'][lang][np.average(convo.out_degree_hist)] += 1

                # x['avg_depth'][lang][np.average(convo.depths)] += 1
                x['tree_depth'][lang][convo.tree_depth] += 1

                # x['avg_width'][lang][np.average(convo.widths)] += 1
                x['tree_width'][lang][convo.tree_width] += 1
                for s in rep_degree_strings:
                    x['degrees'][lang][s] += 1

        for lang in x[keys[0]]:
            out = {k: dict(x[k][lang]) for k in keys}
            os.makedirs(f'out/{args.sel}/{lang}/convo/', exist_ok=True)
            json.dump(out, open(f'out/{args.sel}/{lang}/convo/graph.json', 'w+'))

        return {k: dict(x[k][target]) for k in keys}


def load_mu(target):
    try:
        return json.load(open(f'out/{args.sel}/{target}/convo/mu.json'))
    except FileNotFoundError:
        mus = defaultdict(lambda: defaultdict(int))
        for convo, langs in get_convo_iterator():
            if convo.messages < 2:
                continue

            langs.add('all')
            if 'en' in langs or 'und' in langs:
                langs.add('en_und')

            mu = convo.decay_rate
            if mu is None:
                continue

            for lang in langs:
                mus[lang][mu] += 1

        for lang in tqdm(mus):
            os.makedirs(f'out/{args.sel}/{lang}/post/', exist_ok=True)
            json.dump(dict(mus[lang]), open(f'out/{args.sel}/{lang}/convo/mu.json', 'w+'))

        return dict(mus)[target]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', dest='data', required=True, type=str)
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
    parser.add_argument('--metric', dest='metric', type=str, default='size',
                        const='size',
                        nargs='?',
                        choices=['text', 'size', 'graph', 'mu'])

    args = parser.parse_args()

    # other vars
    data_root = args.data
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

    data = load(target=tgt_langs, metric=args.metric)
