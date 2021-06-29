"""
This file represents an executable that demonstrates
conversion from a raw representation
into a conversation segmented, JSON-line separated output
for further downstream analysis.
"""
import json
import os
from argparse import ArgumentParser

from pyconversations.reader import ChanReader
from pyconversations.reader import QuoteReader
from pyconversations.reader import RawFBReader
from pyconversations.reader import RedditReader
from pyconversations.reader import ThreadsReader


def preprocess_buzzface():
    for pagename, convo_chunk in RawFBReader.iter_read(data_root + 'BuzzFace/data*/'):
        print(f'{pagename}: {len(convo_chunk)} conversations')
        lines = [json.dumps(convo.to_json()) for convo in convo_chunk]

        os.makedirs(out + 'FB/BuzzFace', exist_ok=True)
        with open(out + f'FB/BuzzFace/{pagename}.json', 'w+') as fp:
            fp.write('\n'.join(lines))


def preprocess_outlets():
    for pagename, convo_chunk in RawFBReader.iter_read(data_root + 'Outlets/data*/'):
        print(f'{pagename}: {len(convo_chunk)} conversations')
        lines = [json.dumps(convo.to_json()) for convo in convo_chunk]

        os.makedirs(out + 'FB/Outlets', exist_ok=True)
        with open(out + f'FB/Outlets/{pagename}.json', 'w+') as fp:
            fp.write('\n'.join(lines))


def preprocess_chunked_4chan(board):
    for ix, convo_chunk in ChanReader.iter_read(data_root + f'4chan/{board}/'):
        print(f'{ix}: {len(convo_chunk)} conversations')
        lines = [json.dumps(convo.to_json()) for convo in convo_chunk]

        os.makedirs(out + f'4chan/{board}/', exist_ok=True)
        with open(out + f'4chan/{board}/{ix:02d}.json', 'w+') as fp:
            fp.write('\n'.join(lines))


def pre_process_quote_tweets(sharding=100):
    convo_chunks = QuoteReader.read(data_root + 'quote_tweets/quotes/')

    total = len(convo_chunks)
    print(f'{total} conversations')
    print(f'Sharding into {sharding} chunks with ~{total / sharding} conversations per chunk.')

    os.makedirs(out + 'Twitter/CTQ/', exist_ok=True)
    cap = (total // sharding) + 1
    cache = []
    shard = 0
    for convo in convo_chunks:
        # convo.redact()
        cache.append(json.dumps(convo.to_json()))

        if len(cache) >= cap:
            with open(out + f'Twitter/CTQ/{shard:02d}.json', 'w+') as fp:
                fp.write('\n'.join(cache))
            cache = []
            shard += 1

    if cache:
        with open(out + f'Twitter/CTQ/{shard:02d}.json', 'w+') as fp:
            fp.write('\n'.join(cache))


def preprocess_newstweetthreads(per_file=1_000):
    os.makedirs(out + 'Twitter/NTT/', exist_ok=True)

    write_cache = []
    cnt = 0
    for ix, convo_chunk in ThreadsReader.iter_read(data_root + 'threads/'):
        print(f'{ix}: {len(convo_chunk)} conversations')
        # for chunk in convo_chunk:
        #     chunk.redact()
        write_cache.extend([json.dumps(convo.to_json()) for convo in convo_chunk])

        if len(write_cache) >= per_file:
            with open(out + f'Twitter/NTT/{cnt:04d}.json', 'w+') as fp:
                fp.write('\n'.join(write_cache))
            write_cache = []
            cnt += 1

    if write_cache:
        with open(out + f'Twitter/NTT/{cnt:06d}.json', 'w+') as fp:
            fp.write('\n'.join(write_cache))


def preprocess_reddit_cmv(per_file=1_000):
    os.makedirs(out + 'Reddit/CMV/', exist_ok=True)

    write_cache = []
    cnt = 0
    for convo_chunk in RedditReader.iter_read(data_root + 'cmv-full-2017-09-22/'):
        write_cache.extend([json.dumps(convo.to_json()) for convo in convo_chunk])

        if len(write_cache) >= per_file:
            with open(out + f'Reddit/CMV/{cnt:06d}.json', 'w+') as fp:
                fp.write('\n'.join(write_cache))
            write_cache = []
            cnt += 1

    if write_cache:
        with open(out + f'Reddit/CMV/{cnt:06d}.json', 'w+') as fp:
            fp.write('\n'.join(write_cache))


def preprocess_reddit_dialog(board):
    os.makedirs(out + f'Reddit/RD_{board}/', exist_ok=True)

    cnt = 0
    for convo_chunk in RedditReader.iter_read(data_root + f'raw_rd/[0-9][0-9][0-9][0-9]-[0-9][0-9]_{board}', rd=True):
        write = [json.dumps(convo.to_json()) for convo in convo_chunk]
        with open(out + f'Reddit/RD_{board}/{cnt:03d}.json', 'w+') as fp:
            fp.write('\n'.join(write))
        cnt += 1


if __name__ == '__main__':
    parser = ArgumentParser('Demo executable of how one might read raw data into conversational format.')
    parser.add_argument('--data', dest='data', required=True, type=str, help='General directory data is located in')
    parser.add_argument('--out', dest='out', type=str, default='conversations/')
    parser.add_argument('--sel', dest='sel', type=str, default='bf',
                        const='bf',
                        nargs='?',
                        choices=[
                            'cmv',
                            'rd-politics', 'rd-news', 'rd-worldnews',
                            'ntt', 'ctq',
                            '4chan-news', '4chan-sci', '4chan-his', '4chan-x', '4chan-g', '4chan-pol',
                            'outlets', 'bf'
                        ],
                        help='Dataset key in selection')

    args = parser.parse_args()

    data_root = args.data
    out = data_root + args.out

    if args.sel == 'bf':
        preprocess_buzzface()
    elif args.sel == 'outlets':
        preprocess_outlets()
    elif '4chan' in args.sel:
        board = args.sel.split('-')[-1]
        preprocess_chunked_4chan(board)
    elif args.sel == 'ctq':
        pre_process_quote_tweets()
    elif args.sel == 'ntt':
        preprocess_newstweetthreads()
    elif args.sel == 'cmv':
        preprocess_reddit_cmv()
    elif args.sel[:2] == 'rd':
        board = args.sel.split('-')[-1]
        preprocess_reddit_dialog(board)
