import json
from argparse import ArgumentParser
from collections import Counter
from glob import glob

from pyconversations.message import ChanPost
from pyconversations.reader import ConvoReader
from pyconversations.utils import num2str


def get_post_iterator(print_every=100_000):
    cnt = 0
    for convo in ConvoReader.iter_read(data_root + dataset, cons=cons):
        for post in convo.posts.values():
            if cnt and cnt % print_every == 0:
                print(f'Post {num2str(cnt)}')
            cnt += 1

            yield post


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', dest='data', required=True, type=str)

    args = parser.parse_args()

    data_root = args.data

    try:
        print(json.load(open('out/anon_ratios.json')))
    except FileNotFoundError:
        cons = ChanPost
        title = '4Chan'
        anon_count = Counter()
        total_count = Counter()

        for f in glob(data_root + '4chan/*/'):
            board = f.split('/')[-2]
            dataset = '4chan/' + board + '/'

            print(dataset)
            for post in get_post_iterator():
                total_count[board] += 1
                total_count['all'] += 1

                if post.author == 'Anonymous':
                    anon_count[board] += 1
                    anon_count['all'] += 1

        ratios = {k: anon_count[k] / total_count[k] for k in total_count}
        json.dump(ratios, open('out/anon_ratios.json', 'w+'))
        print(ratios)
