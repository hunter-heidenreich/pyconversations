import json
from glob import glob

from tqdm import tqdm

from pyconversations.convo import Conversation
from pyconversations.message import RedditPost
from pyconversations.message import Tweet


def read_tweets(path):
    return Tweet.parse_raw(json.load(open(path)), lang_detect=True)


def read_reddit_posts(path):
    raw = json.load(open(path))
    if type(raw) == dict:
        return RedditPost.parse_raw(raw, lang_detect=True)
    else:
        return [y for x in raw for y in RedditPost.parse_raw(x, lang_detect=True)]


def load_rumoureval(data_path):
    # Answer keys, we can use these to extract some annotations
    # and tag data with their associated split too\
    train_path = data_path + 'rumoureval-2019-training-data/'
    test_path = data_path + 'rumoureval-2019-test-data/'
    TRAIN_KEY = json.load(open(train_path + 'train-key.json'))
    DEV_KEY = json.load(open(train_path + 'dev-key.json'))
    FINAL = json.load(open(data_path + 'final-eval-key.json'))

    TRAIN_KEY.keys(), DEV_KEY.keys(), FINAL.keys()

    data_set = Conversation()

    # reading Twitter data
    for tweet_path in tqdm(
            [
                x
                for pat in ('source-tweet', 'replies')
                for x in glob(train_path + f'twitter-english/*/*/{pat}/*.json')
            ], desc='reading train+dev'):

        for t in read_tweets(tweet_path):
            subject = tweet_path.split('/')[-3]
            tstr = str(t.uid)
            if tstr in TRAIN_KEY['subtaskaenglish']:
                t.add_tag('split=TRAIN')
                t.add_tag(f'taskA={TRAIN_KEY["subtaskaenglish"][tstr]}')
                if tstr in TRAIN_KEY['subtaskbenglish']:
                    t.add_tag(f'taskB={TRAIN_KEY["subtaskbenglish"][tstr]}')
            elif tstr in DEV_KEY['subtaskaenglish']:
                t.add_tag('split=DEV')
                t.add_tag(f'taskA={DEV_KEY["subtaskaenglish"][tstr]}')
                if tstr in DEV_KEY['subtaskbenglish']:
                    t.add_tag(f'taskB={DEV_KEY["subtaskbenglish"][tstr]}')
            elif tstr in FINAL['subtaskaenglish']:
                t.add_tag('split=TEST')
                t.add_tag(f'taskA={FINAL["subtaskaenglish"][tstr]}')
                if tstr in FINAL['subtaskbenglish']:
                    t.add_tag(f'taskB={FINAL["subtaskbenglish"][tstr]}')

            t.add_tag(subject)

            data_set.add_post(t)

    for tweet_path in tqdm(
            [
                x
                for pat in ('source-tweet', 'replies')
                for x in glob(test_path + f'twitter-en-test-data/*/*/{pat}/*.json')
            ], desc='reading test'):
        for t in read_tweets(tweet_path):
            subject = tweet_path.split('/')[-3]
            tstr = str(t.uid)
            if tstr in FINAL['subtaskaenglish']:
                t.add_tag('split=TEST')
                t.add_tag(f'taskA={FINAL["subtaskaenglish"][tstr]}')
                if tstr in FINAL['subtaskbenglish']:
                    t.add_tag(f'taskB={FINAL["subtaskbenglish"][tstr]}')

            t.add_tag(subject)

            data_set.add_post(t)

    # reading Reddit data
    for reddit_path in tqdm(
        list(glob(train_path + 'reddit-training-data/*/raw.json')) +
        [x for pat in ('source-tweet', 'replies') for x in glob(train_path + f'reddit-training-data/*/{pat}/*.json')],
        desc='reading train'
    ):
        for t in read_reddit_posts(reddit_path):
            t.add_tag('split=TRAIN')
            if t.uid in TRAIN_KEY['subtaskaenglish']:
                t.add_tag(f'taskA={TRAIN_KEY["subtaskaenglish"][t.uid]}')

            if t.uid in TRAIN_KEY['subtaskbenglish']:
                t.add_tag(f'taskB={TRAIN_KEY["subtaskbenglish"][t.uid]}')

            data_set.add_post(t)

    for reddit_path in tqdm(
        list(glob(train_path + 'reddit-dev-data/*/raw.json')) +
        [x for pat in ('source-tweet', 'replies') for x in glob(train_path + f'reddit-dev-data/*/{pat}/*.json')],
        desc='reading dev'
    ):
        for t in read_reddit_posts(reddit_path):
            t.add_tag('split=DEV')

            if t.uid in DEV_KEY['subtaskaenglish']:
                t.add_tag(f'taskA={DEV_KEY["subtaskaenglish"][t.uid]}')

            if t.uid in DEV_KEY['subtaskbenglish']:
                t.add_tag(f'taskB={DEV_KEY["subtaskbenglish"][t.uid]}')

            data_set.add_post(t)

    for reddit_path in tqdm(
        list(glob(train_path + 'reddit-test-data/*/raw.json')) +
        [x for pat in ('source-tweet', 'replies') for x in glob(test_path + f'reddit-test-data/*/{pat}/*.json')],
        'reading test'
    ):
        for t in read_reddit_posts(reddit_path):
            t.add_tag('split=TEST')

            if t.uid in FINAL['subtaskaenglish']:
                t.add_tag(f'taskA={FINAL["subtaskaenglish"][t.uid]}')

            if t.uid in FINAL['subtaskbenglish']:
                t.add_tag(f'taskB={FINAL["subtaskbenglish"][t.uid]}')

            data_set.add_post(t)

    return data_set


if __name__ == '__main__':
    # Update this to your local data directory
    DATA_PATH = '/Users/hsh28/data/rumoureval2019/'
