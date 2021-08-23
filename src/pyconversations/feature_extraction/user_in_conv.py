from collections import Counter
from collections import defaultdict
from functools import lru_cache
from functools import reduce

import numpy as np

from ..convo import Conversation
from .harmonic import mixing
from .harmonic import novelty
from .params import CACHE_SIZE
from .post import type_frequency_distribution as post_freq
from .post_in_conv import PostInConvoFeatures as PICF
from .post_in_conv import agg_post_stats
from .post_in_conv import avg_token_entropy_conv


class UserInConvoFeatures:
    """
    Container for feature extraction on users situated within a single conversation.
    """

    @staticmethod
    def bools(user, convo):
        return {
            'is_source_author': is_source_author(user, convo),
        }

    @staticmethod
    def categoricals(user, convo):
        return {

        }

    @staticmethod
    def counter(user, convo):
        return {
            'type_freq': type_frequency_distribution(user, convo),
        }

    @staticmethod
    def floats(user, convo):
        out = mixing_features(user, convo)
        out['avg_user_token_entropy'] = avg_user_token_entropy(user, convo)

        for k, v in agg_post_stats(convo, filter_by=lambda p: p.author == user).items():
            out[k] = v

        return out

    @staticmethod
    def ints(user, convo):
        out = {
            'message_count': messages_by_user(user, convo),
            'types':         len(type_frequency_distribution(user, convo)),
        }

        for k, v in sum_post_bools_by_user(user, convo).items():
            out[k] = v

        for k, v in sum_post_ints_by_user(user, convo).items():
            out[k] = v

        return out

    @staticmethod
    def strs(user, convo):
        return {}


def iter_over_users(convo):
    """
    Creates an iterator over the unique users within a conversation

    Parameters
    ----------
    convo : Conversation

    Yields
    -------
    str
        A user
    """
    users = set()
    for pid in convo.posts:
        user = convo.posts[pid].author

        if user in users:
            continue

        yield user

        users.add(user)


@lru_cache(maxsize=CACHE_SIZE)
def is_source_author(user, convo):
    """
    Returns if this user created a source message
    for the conversation

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    bool
    """
    auths = {convo.posts[pid].author for pid in convo.get_sources()}
    return user in auths


@lru_cache(maxsize=CACHE_SIZE)
def messages_per_user(conv):
    """
    Returns the user-distribution of posts written per user

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        The counts of messages written per user (keyed by author name)
    """
    return Counter([p.author for p in conv.posts.values()])


@lru_cache(maxsize=CACHE_SIZE)
def messages_by_user(user, conv):
    """
    Returns number of messages created by this user
    Parameters
    ----------
    user : str
    conv : Conversation

    Returns
    -------
    int
    """
    return messages_per_user(conv)[user]


@lru_cache(maxsize=CACHE_SIZE)
def get_user_posts(user, conv):
    """
    Filters to just this users messages within the conversation

    Parameters
    ----------
    user : str
    conv : Conversation

    Returns
    -------
    Conversation
    """
    if user is None:
        return Conversation()
    return Conversation(posts={pid: conv.posts[pid] for pid in conv.filter(by_author=user)},
                        convo_id=f'{conv.convo_id}-{user}')


@lru_cache(maxsize=CACHE_SIZE)
def type_frequency_distribution(user, convo):
    """
    Returns the type frequency (unigram) distribution for the user's posts in the convo.

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    collections.Counter
    """
    return reduce(lambda x, y: x + y, map(post_freq, get_user_posts(user, convo).posts.values()), Counter())


@lru_cache(maxsize=CACHE_SIZE)
def mixing_features(user, convo):
    """
    Returns the measured parameters using the harmonic mixing law.

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    dict(str, float)
    """
    freq = type_frequency_distribution(user, convo)
    if len(freq) == 0:
        return {
            'k1':      float(0),
            'theta':   float(0),
            'entropy': float(0),
            'N_avg':   float(0),
            'M_avg':   float(0),
        }

    return mixing(freq)


@lru_cache(maxsize=CACHE_SIZE)
def novelty_vector(user, convo):
    """
    Returns the novelty vector measured from the convo text, filtered to just the user.

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    np.array
    """
    freq = type_frequency_distribution(user, convo)
    if len(freq) == 0:
        return []

    return novelty(freq)


@lru_cache(maxsize=CACHE_SIZE)
def avg_user_token_entropy(user, convo):
    """
    Returns the average token entropy when comparing a user
    to a conversation (containing said user's posts)

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    float
    """
    return float(avg_token_entropy_conv(get_user_posts(user, convo), convo))


def agg_user_stats(convo):
    """
    Computes a set of aggregate user statistical measures.
    This is only computed for the integer and float subsets.
    Specifically, the following stats are measured:
    min, max, mean, median, standard deviation


    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    dict(str, dict(str, float))
    """
    agg = defaultdict(list)
    fs = [UserInConvoFeatures.floats, UserInConvoFeatures.ints]
    for user in iter_over_users(convo):
        for f in fs:
            for k, v in f(user, convo).items():
                agg[k].append(v)

    out = {}
    for k, vs in agg.items():
        out[f'user_min_{k}'] = float(np.nanmin(vs))
        out[f'user_max_{k}'] = float(np.nanmax(vs))
        out[f'user_mean_{k}'] = float(np.nanmean(vs))
        out[f'user_median_{k}'] = float(np.median(vs))
        out[f'user_std_{k}'] = float(np.nanstd(vs) if len(vs) > 1 else 1)

    return out


def sum_post_bools_by_user(user, convo):
    cnt = Counter()

    for p in convo.posts.values():
        if p.author != user:
            continue

        for k, v in PICF.bools(p, convo).items():
            kx = k.replace('is_', '') + '_count'
            cnt[kx] += 1 if v else 0

    return dict(cnt)


def sum_post_ints_by_user(user, convo):
    skipset = {
        'type_count',  # must be aggregated in set theoretic way
        'depth', 'width',  # nonsensical accumulation stats
    }
    cnt = Counter()
    for p in convo.posts.values():
        if p.author != user:
            continue

        for k, v in PICF.ints(p, convo).items():
            if k in skipset:
                continue

            cnt[k] += v

    return dict(cnt)
