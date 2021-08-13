from collections import Counter
from functools import lru_cache
from functools import reduce

import networkx as nx

from .harmonic import mixing
from .harmonic import novelty
from .params import CACHE_SIZE
from .post import get_all as post_get_all
from .post import type_frequency_distribution as post_freq
from .post_in_conv import agg_post_stats
from .post_in_conv import depth_dist
from .post_in_conv import get_all as pic_get_all
from .post_in_conv import sum_booleans_across_convo as sum_post_bools
from .post_in_conv import sum_ints_across_convo as sum_post_ints
from .user_in_conv import messages_per_user
from .utils import apply_extraction


def get_all(cx, keys=None, ignore_keys=None, include_post=True):
    """
    Returns all features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)
    include_post : bool

    Returns
    -------
    dict(str, Any)
    """
    out = {
        **get_counters(cx, keys, ignore_keys),
        **get_floats(cx, keys, ignore_keys),
        **get_ints(cx, keys, ignore_keys),
    }

    if include_post:
        out = {**agg_post_stats(cx, keys, ignore_keys), **out}

    return out


def get_counters(cx, keys=None, ignore_keys=None):
    """
    Returns all Counter-features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, Counter)
    """
    return apply_extraction({
        'degree_size_distribution':     degree_size_distribution,
        'degree_in_size_distribution':  degree_in_size_distribution,
        'degree_out_size_distribution': degree_out_size_distribution,
        'depth_distribution':           depth_dist,
        'type_frequency_distribution':  type_frequency_distribution,
        'user_size_distribution':       user_size_dist,
    }, keyset=keys, ignore=ignore_keys, convo=cx)


def get_floats(cx, keys=None, ignore_keys=None):
    """
    Returns all integer features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, int)
    """
    return {
        **apply_extraction({
            'density':        density,
            'duration':       duration,
            'mixing_k1': lambda convo: mixing_features(convo)['k1'],
            'mixing_theta': lambda convo: mixing_features(convo)['theta'],
            'mixing_entropy': lambda convo: mixing_features(convo)['entropy'],
            'mixing_N_avg': lambda convo: mixing_features(convo)['N_avg'],
            'mixing_M_avg': lambda convo: mixing_features(convo)['M_avg'],
        }, keyset=keys, ignore=ignore_keys, convo=cx),
    }


def get_ints(cx, keys=None, ignore_keys=None):
    """
    Returns all integer features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, int)
    """
    return {
        **apply_extraction({
            'messages': lambda convo: len(convo.posts),
            'tree_degree': tree_degree,
            'tree_depth':  tree_depth,
            'tree_width':  tree_width,
            'types': lambda convo: len(type_frequency_distribution(convo)),
            'users': lambda convo: len(messages_per_user(convo)),
        }, keyset=keys, ignore=ignore_keys, convo=cx),
        **sum_post_bools(cx),
        **sum_post_ints(cx),
    }


@lru_cache(maxsize=CACHE_SIZE)
def degree_size_distribution(convo):
    """
    Returns the post degree size distribution for this Conversation.
    Keys indicate a post degree and values indicate the number of posts with that degree
    within this Conversation.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    collections.Counter
    """
    return Counter([pic_get_all(p, convo, keys={'degree'})['degree'] for p in convo.posts.values()])


@lru_cache(maxsize=CACHE_SIZE)
def degree_in_size_distribution(convo):
    """
    Returns the post in-degree size distribution for this Conversation.
    Keys indicate a post degree and values indicate the number of posts with that degree
    within this Conversation.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    collections.Counter
    """
    return Counter([pic_get_all(p, convo, keys={'degree_in'})['degree_in'] for p in convo.posts.values()])


@lru_cache(maxsize=CACHE_SIZE)
def degree_out_size_distribution(convo):
    """
    Returns the post out-degree size distribution for this Conversation.
    Keys indicate a post degree and values indicate the number of posts with that degree
    within this Conversation.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    collections.Counter
    """
    return Counter([post_get_all(p, keys={'degree_out'})['degree_out'] for p in convo.posts.values()])


@lru_cache(maxsize=CACHE_SIZE)
def user_size_dist(conv):
    """
    Returns a distribution of the number of posts per user mapping to the number of users
    that contributed that many posts to `conv`.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        The size distribution mapping from (# of posts) -> (# of users that added that many posts to `conv`)
    """
    return Counter(list(messages_per_user(conv).values()))


@lru_cache(maxsize=CACHE_SIZE)
def density(conv):
    """
    The density of the conversation as a DAG

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    float
        The density of connection with in the conversation

    Notes
    -----
    See for more information: https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.density.html
    """
    return nx.density(conv.as_graph())


@lru_cache(maxsize=CACHE_SIZE)
def tree_depth(conv):
    """
    Returns the depth of the full conversation.
    This is the max depth of any post within the Conversation.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    int
        Depth of the entire conversation as a DAG
    """
    return max(depth_dist(conv).keys())


@lru_cache(maxsize=CACHE_SIZE)
def tree_width(conv):
    """
    Returns the width of the full conversation.
    This is the max width (# of posts) at any depth level within the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    int
        Width of the entire conversation as a DAG
    """
    return max(depth_dist(conv).values())


@lru_cache(maxsize=CACHE_SIZE)
def tree_degree(conv):
    """
    Returns the degree of the full conversation.
    This is the max degree of any post within it

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    int
        Degree of the entire conversation as a DAG
    """
    return max(degree_size_distribution(conv).keys())


@lru_cache(maxsize=CACHE_SIZE)
def time_series(conv, normalize_by_first=False):
    """
    Returns the list of timestamps of when posts where added to this Conversation.
    If `normalize_by_first`, then all timestamps are reduced by the start time of the first message.

    Parameters
    ----------
    conv : Conversation
    normalize_by_first : bool
        Default: False

    Returns
    -------
    list(float)
    """
    order = conv.time_order()
    out = [conv.posts[uid].created_at.timestamp() for uid in order] if order else []
    if normalize_by_first:
        start = out[0]
        out = [o - start for o in out]

    return out


@lru_cache(maxsize=CACHE_SIZE)
def duration(conv):
    """
    Returns the length of the converation in seconds.

    Parameters
    ----------
    conv : Conversation

    Returns
    -------
    float
    """
    ts = time_series(conv)
    return ts[-1] - ts[0]


@lru_cache(maxsize=CACHE_SIZE)
def type_frequency_distribution(convo):
    """
    Returns the type frequency (unigram) distribution for the convo.

    Parameters
    ----------
    convo : Convrersation

    Returns
    -------
    collections.Counter
    """
    return reduce(lambda x, y: x + y, map(post_freq, convo.posts.values()))


@lru_cache(maxsize=CACHE_SIZE)
def mixing_features(convo):
    """
    Returns the measured parameters using the harmonic mixing law.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    dict(str, float)
    """
    return mixing(type_frequency_distribution(convo))


@lru_cache(maxsize=CACHE_SIZE)
def novelty_vector(convo):
    """
    Returns the novelty vector measured from the convo text.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    np.array
    """
    return novelty(type_frequency_distribution(convo))
