import os

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from pyconversations.message import *
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

    args = parser.parse_args()

    data_root = args.data
    os.makedirs('out/convo/', exist_ok=True)

    if args.sel == 'bf':
        dataset = 'BuzzFace/'
        cons = FBPost
        title = 'BuzzFace'
    elif args.sel == 'outlets':
        dataset = 'Outlets/'
        cons = FBPost
        title = 'Outlets'
    elif args.sel == 'chan':
        dataset = '4chan/*/'
        cons = ChanPost
        title = '4Chan'
    elif '4chan' in args.sel:
        dataset = args.ds.replace('-', '/') + '/'
        cons = ChanPost
        title = dataset.replace('4chan', '')
    elif args.sel == 'ctq':
        dataset = 'CTQuotes/'
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

    cnt = 0
    print_every = 2_000

    df = []
    for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
        metric = {
            # general
            'messages': convo.messages,
            'connections': convo.connections,
            'users': convo.users,
            'duration': convo.duration,

            # text
            'chars': convo.chars,
            'tokens': convo.tokens,
            'types': len(convo.token_types),

            # graph
            'density': convo.density,

            'avg_degree': np.mean(convo.degree_hist),
            'std_degree': np.std(convo.degree_hist),
            'avg_in_degree': np.mean(convo.in_degree_hist),
            'std_in_degree': np.std(convo.in_degree_hist),
            'avg_out_degree': np.mean(convo.out_degree_hist),
            'std_out_degree': np.std(convo.out_degree_hist),

            'avg_depth': np.mean(convo.depths),
            'std_depth': np.std(convo.depths),
            'depth': convo.tree_depth,
            'avg_width': np.mean(convo.widths),
            'std_width': np.std(convo.widths),
            'width': convo.tree_width,

            'diameter': convo.diameter,
            'radius': convo.radius,

            'assortativity': convo.assortativity,
            'rich_club_coef': convo.rich_club_coefficient
        }
        df.append(metric)

        cnt += 1
        if cnt % print_every == 0:
            print(f'Processed {cnt} conversations.')

    df = pd.DataFrame(df)
    df.to_csv(f'out/convo/{args.sel}.csv')
