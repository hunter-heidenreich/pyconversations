"""
This file represents an executable that demonstrates
conversion from a raw representation
into a conversation segmented, JSON-line separated output
for further downstream analysis.
"""
import json
import os

from argparse import ArgumentParser
from pyconversations.reader import RawFBReader, ChanReader, QuoteReader, ThreadsReader, RedditReader


def preprocess_buzzface():
    for pagename, convo_chunk in RawFBReader.iter_read(data_root + 'BuzzFace/data*/'):
        print(f'{pagename}: {len(convo_chunk)} conversations')
        lines = [json.dumps(convo.to_json()) for convo in convo_chunk]

        os.makedirs(out + 'BuzzFace', exist_ok=True)
        with open(out + f'BuzzFace/{pagename}.json', 'w+') as fp:
            fp.write('\n'.join(lines))


def preprocess_outlets():
    for pagename, convo_chunk in RawFBReader.iter_read(data_root + 'Outlets/data*/'):
        print(f'{pagename}: {len(convo_chunk)} conversations')
        lines = [json.dumps(convo.to_json()) for convo in convo_chunk]

        os.makedirs(out + 'Outlets', exist_ok=True)
        with open(out + f'Outlets/{pagename}.json', 'w+') as fp:
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

    os.makedirs(out + f'CTQuotes/', exist_ok=True)
    cap = (total // sharding) + 1
    cache = []
    shard = 0
    for convo in convo_chunks:
        cache.append(json.dumps(convo.to_json()))

        if len(cache) >= cap:
            with open(out + f'CTQuotes/{shard:02d}.json', 'w+') as fp:
                fp.write('\n'.join(cache))
            cache = []
            shard += 1

    if cache:
        with open(out + f'CTQuotes/{shard:02d}.json', 'w+') as fp:
            fp.write('\n'.join(cache))


def preprocess_newstweetthreads(per_file=1_000):
    os.makedirs(out + f'threads/', exist_ok=True)

    write_cache = []
    cnt = 0
    for ix, convo_chunk in ThreadsReader.iter_read(data_root + f'threads/'):
        print(f'{ix}: {len(convo_chunk)} conversations')
        write_cache.extend([json.dumps(convo.to_json()) for convo in convo_chunk])

        if len(write_cache) >= per_file:
            with open(out + f'threads/{cnt:06d}.json', 'w+') as fp:
                fp.write('\n'.join(write_cache))
            write_cache = []
            cnt += 1

    if write_cache:
        with open(out + f'threads/{cnt:06d}.json', 'w+') as fp:
            fp.write('\n'.join(write_cache))


def preprocess_reddit_cmv(per_file=1_000):
    os.makedirs(out + f'Reddit/CMV/', exist_ok=True)

    write_cache = []
    cnt = 0
    for convo_chunk in RedditReader.iter_read(data_root + f'cmv-full-2017-09-22/'):
        write_cache.extend([json.dumps(convo.to_json()) for convo in convo_chunk])

        if len(write_cache) >= per_file:
            with open(out + f'Reddit/CMV/{cnt:06d}.json', 'w+') as fp:
                fp.write('\n'.join(write_cache))
            write_cache = []
            cnt += 1

    if write_cache:
        with open(out + f'Reddit/CMV/{cnt:06d}.json', 'w+') as fp:
            fp.write('\n'.join(write_cache))


if __name__ == '__main__':
    parser = ArgumentParser('Demo executable of how one might read raw data into conversational format.')
    parser.add_argument('--data', dest='data', type=str, help='General directory data is located in')

    args = parser.parse_args()

    data_root = args.data
    out = data_root + 'conversations/'

    # preprocess_buzzface()
    # preprocess_outlets()

    # preprocess_chunked_4chan('news')
    # preprocess_chunked_4chan('sci')
    # preprocess_chunked_4chan('his')
    # preprocess_chunked_4chan('x')
    # preprocess_chunked_4chan('g')
    # preprocess_chunked_4chan('pol')

    # pre_process_quote_tweets()
    # preprocess_newstweetthreads()

    preprocess_reddit_cmv()
