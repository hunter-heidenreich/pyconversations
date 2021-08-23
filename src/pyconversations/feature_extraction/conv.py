from collections import Counter
from collections import defaultdict
from functools import lru_cache

import networkx as nx
import numpy as np

from .harmonic import mixing
from .harmonic import novelty
from .params import CACHE_SIZE
from .post_in_conv import agg_post_stats
from .post_in_conv import conversation_type_frequency_distribution as type_frequency_distribution
from .post_in_conv import depth_dist
from .post_in_conv import out_degree
from .post_in_conv import post_degree
from .post_in_conv import post_in_degree
from .post_in_conv import sum_booleans_across_convo as sum_post_bools
from .post_in_conv import sum_ints_across_convo as sum_post_ints
from .user_in_conv import agg_user_stats
from .user_in_conv import messages_per_user


class ConvoFeatures:

    """
    Container of feature extraction for conversations
    """

    @staticmethod
    def bools(convo):
        return {}

    @staticmethod
    def categoricals(convo):
        return {}

    @staticmethod
    def counter(convo):
        return {
            'degree_size_distribution':     degree_size_distribution(convo),
            'degree_in_size_distribution':  degree_in_size_distribution(convo),
            'degree_out_size_distribution': degree_out_size_distribution(convo),
            'depth_distribution':           depth_dist(convo),
            'type_frequency_distribution':  type_frequency_distribution(convo),
            'user_size_distribution':       user_size_dist(convo),
        }

    @staticmethod
    def floats(convo):
        out = mixing_features(convo)
        out['duration'] = duration(convo)
        out['density'] = density(convo)

        for k, v in agg_post_stats(convo).items():
            out[k] = v

        for k, v in agg_user_stats(convo).items():
            out[k] = v

        return out

    @staticmethod
    def ints(convo):
        out = {
            'messages':    len(convo.posts),
            'tree_degree': tree_degree(convo),
            'tree_depth':  tree_depth(convo),
            'tree_width':  tree_width(convo),
            'types':       len(type_frequency_distribution(convo)),
            'users':       len(messages_per_user(convo)),
        }

        for k, v in sum_post_bools(convo).items():
            out[k] = v

        for k, v in sum_post_ints(convo).items():
            out[k] = v

        return out

    @staticmethod
    def strs(convo):
        return {}


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
    return Counter([post_degree(p, convo) for p in convo.posts.values()])


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
    return Counter([post_in_degree(p, convo) for p in convo.posts.values()])


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
    return Counter([out_degree(p) for p in convo.posts.values()])


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
def time_series(conv, normalize_by_first=True):
    """
    Returns the list of timestamps of when posts where added to this Conversation.
    If `normalize_by_first`, then all timestamps are reduced by the start time of the first message.

    Parameters
    ----------
    conv : Conversation
    normalize_by_first : bool
        Default: True

    Returns
    -------
    list(float)
    """
    order = conv.time_order()
    out = [conv.posts[uid].created_at.timestamp() for uid in order] if order else []
    if normalize_by_first and out:
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

    if not ts:
        return -1

    return ts[-1] - ts[0]


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
    freq = type_frequency_distribution(convo)
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
    freq = type_frequency_distribution(convo)

    if len(freq) == 0:
        return []

    return novelty(freq)


def agg_convo_stats(convos):
    """
    Computes a set of aggregate conversation statistical measures.
    This is only computed for the integer and float subsets.
    Specifically, the following stats are measured:
    min, max, mean, median, standard deviation


    Parameters
    ----------
    convos : List(Conversation)

    Returns
    -------
    dict(str, dict(str, float))
    """
    agg = defaultdict(list)
    fs = [ConvoFeatures.floats, ConvoFeatures.ints]
    for conv in convos:
        for f in fs:
            for k, v in f(conv).items():
                agg[k].append(v)

    out = {}
    for k, vs in agg.items():
        out[f'convo_min_{k}'] = float(np.nanmin(vs))
        out[f'convo_max_{k}'] = float(np.nanmax(vs))
        out[f'convo_mean_{k}'] = float(np.nanmean(vs))
        out[f'convo_median_{k}'] = float(np.median(vs))
        out[f'convo_std_{k}'] = float(np.nanstd(vs) if len(vs) > 1 else 1)

    return out
