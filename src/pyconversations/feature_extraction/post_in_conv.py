from collections import Counter
from collections import defaultdict
from functools import lru_cache
from functools import reduce

import numpy as np

from ..convo import Conversation
from .params import CACHE_SIZE
from .post import PostFeatures
from .post import is_source
from .post import out_degree
from .post import type_frequency_distribution as post_freq


class PostInConvoFeatures:

    """
    Container of feature extraction for posts in the conversation they appeared in.
    Features for posts in isolation are nested within this extraction.
    """

    @staticmethod
    def bools(post, convo):
        out = PostFeatures.bools(post)
        out['is_leaf'] = is_leaf(post, convo)
        out['is_internal'] = is_internal(post, convo)
        out['is_author_source_author'] = is_author_source_author(post, convo)
        return out

    @staticmethod
    def categoricals(post, convo):
        out = PostFeatures.categoricals(post)

        return out

    @staticmethod
    def counter(post, convo):
        out = PostFeatures.counter(post)

        return out

    @staticmethod
    def floats(post, convo):
        out = PostFeatures.floats(post)
        out['relative_age'] = post_to_source(post, convo)
        out['response_time'] = post_reply_time(post, convo)

        for k, v in avg_token_entropy_all_splits(post, convo).items():
            out[k] = v

        return out

    @staticmethod
    def ints(post, convo):
        out = PostFeatures.ints(post)
        out['degree'] = post_depth(post, convo)
        out['in_degree'] = post_in_degree(post, convo)
        out['depth'] = post_depth(post, convo)
        out['width'] = post_width(post, convo)

        return out

    @staticmethod
    def strs(post, convo):
        out = PostFeatures.strs(post)

        return out


def is_leaf(post, convo):
    """
    Returns a bool indicating if this post is a leaf within this conversation

    Parameters
    ----------
    post : UniMessage
    convo : Conversation

    Returns
    -------
    bool
    """
    return in_degrees_by_uid(convo)[post.uid] == 0


def is_internal(post, convo):
    """
    Returns a bool indicating if this post is an internal node within this conversation

    Parameters
    ----------
    post : UniMessage
    convo : Conversation

    Returns
    -------
    bool
    """
    return in_degrees_by_uid(convo)[post.uid] != 0 and not is_source(post)


def is_author_source_author(post, convo):
    """
    Returns a bool indicating whether the author of this post is the author
    of a source post within the conversation as well

    Parameters
    ----------
    post : UniMessage
    convo : Conversation

    Returns
    -------
    bool
    """
    return post.author in source_authors(convo)


def post_degree(post, convo):
    return in_degrees_by_uid(convo)[post.uid] + out_degree(post)


def post_in_degree(post, convo):
    return in_degrees_by_uid(convo)[post.uid]


@lru_cache(maxsize=CACHE_SIZE)
def in_degrees_by_uid(conv):
    """
    Returns a Counter of the post IDs mapping to the # of replies that post received in this Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        A mapping from post IDs to the # of replies they receive in `conv`
    """
    cnt = Counter()
    for p in conv.posts.values():
        if p.uid not in cnt:
            cnt[p.uid] = 0

        for r in p.reply_to:
            if r in conv.posts:
                cnt[r] += 1
    return cnt


@lru_cache(maxsize=CACHE_SIZE)
def source_authors(conv):
    """
    Returns the set of authors that contributed a source (non-reply) post.

    Parameters
    ----------
    conv : Conversation

    Returns
    -------
    set(str)
    """
    return set([conv.posts[pid].author for pid in conv.get_sources()])


@lru_cache(maxsize=CACHE_SIZE)
def post_reply_time(post, conv):
    """
    Returns the time between the post and its parent

    Parameters
    ----------
    post : UniMessage
        The message

    conv : Conversation
        A collection of posts

    Returns
    -------
    float
       The time between the `post` and its parent. If multiple parents, returns the minimum response difference
    """
    if post.created_at is None:
        return -1

    diffs = [
        abs((post.created_at - conv.posts[rid].created_at).total_seconds())
        for rid in post.reply_to if rid in conv.posts and conv.posts[rid].created_at is not None
    ]

    if not diffs:
        return -1

    return min(diffs)


@lru_cache(maxsize=CACHE_SIZE)
def post_to_source(post, conv):
    """
    Returns the time between the post and the conversation source

    Parameters
    ----------
    post : UniMessage
        The message

    conv : Conversation
        A collection of posts

    Returns
    -------
    float
       The time between the `post` and its parent. If multiple sources, returns the maximum response difference
    """
    timeorder = conv.time_order()

    if not timeorder:
        return -1

    if post.created_at is None or conv.posts[timeorder[0]].created_at is None:
        return -1

    return (post.created_at - conv.posts[timeorder[0]].created_at).total_seconds()


@lru_cache(maxsize=CACHE_SIZE)
def conversation_type_frequency_distribution(convo):
    """
    Returns the type frequency (unigram) distribution for the convo.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    collections.Counter
    """
    return reduce(lambda x, y: x + y, map(post_freq, convo.posts.values()))


@lru_cache(maxsize=CACHE_SIZE)
def avg_token_entropy(post, conv):
    """
    Returns the average per token normed entropy with respect to the conversation.

    Parameters
    ----------
    post : UniMessage
        The post

    conv : Conversation
        The collection of posts

    Returns
    -------
    float
        The entropy
    """
    if post.uid not in conv.posts:
        cx = Conversation(posts=conv.posts)
        cx.add_post(post)
        conv = cx

    post_dist = post_freq(post)
    convo_dist = conversation_type_frequency_distribution(conv)

    conv_n = len(convo_dist)
    conv_m = sum(convo_dist.values())
    post_m = sum(post_dist.values())

    if not post_m:
        return 0

    if conv_n < 2:
        return 0

    if not conv_m:
        return 0

    numer = np.log([convo_dist[w] / conv_m for w in post_dist])
    denom = (np.log(conv_n) * post_m)
    entropy = -(numer / denom).sum()

    return entropy


@lru_cache(maxsize=CACHE_SIZE)
def avg_token_entropy_conv(conv_a, conv_b):
    """
    Returns the average per token normed entropy of `conv_a` (the first conversation)
    with respect to the joint conversation (addition of `conv_a` and `conv_b`).

    Parameters
    ----------
    conv_a : Conversation
        The collection of posts

    conv_b : Conversation
        The collection of posts

    Returns
    -------
    float
        The entropy
    """
    # Nothing to compare if no posts...
    if not len(conv_a.posts) or not len(conv_b.posts):
        return 0

    joint_conv = conv_a + conv_b

    # Nothing to compare if their identical
    if len(joint_conv.posts) == len(conv_a.posts):
        return 0

    joint_dist = conversation_type_frequency_distribution(joint_conv)

    joint_n = len(joint_dist)
    joint_m = sum(joint_dist.values())

    # Nothing to compare if no types or tokens
    if not joint_m or joint_n < 2:
        return 0

    left_dist = conversation_type_frequency_distribution(conv_a)
    left_m = sum(left_dist.values())

    # Nothing to compare if no tokens
    if not left_m:
        return 0

    numer = np.log([joint_dist[w] / joint_m for w in left_dist])
    denom = (np.log(joint_n) * left_m)
    entropy = -(numer / denom).sum()

    return entropy


@lru_cache(maxsize=CACHE_SIZE)
def avg_token_entropy_all_splits(post, conv):
    splits = {
        'post':        post,
        'full':        conv,

        'ancestors':   conv.get_ancestors(post.uid, include_post=True),
        # 'after':       conv.get_after(post.uid, include_post=True),
        # 'before':      conv.get_before(post.uid, include_post=True),
        'children':    conv.get_children(post.uid, include_post=True),
        'descendants': conv.get_descendants(post.uid, include_post=True),
        'parents':     conv.get_parents(post.uid, include_post=True),
        'siblings':    conv.get_siblings(post.uid, include_post=True),
    }
    """
    Returns a dictionary of average per token normed entropy between
    conversational splits based on `post`.

    Parameters
    ----------
    post : UniMessage
        The post

    conv : Conversation
        The collection of posts

    Returns
    -------
    dict(str, float)
        Mapping from a conversation split pair considered to the entropy measured
    """
    ks = sorted(splits.keys())
    entropy = {}
    for ix, ko in enumerate(ks):
        for iy, ki in enumerate(ks):
            # skip comparison of equal...
            # would always be 1
            if ix == iy:
                continue

            # restrict post to first key only
            if ki == 'post':
                continue

            key = f'{ko}-{ki}'
            if ko == 'post':
                e = avg_token_entropy(post=splits[ko], conv=splits[ki])
            else:
                e = avg_token_entropy_conv(conv_a=splits[ko], conv_b=splits[ki])

            entropy['avg_token_entropy_' + key] = float(e)

    return entropy


@lru_cache(maxsize=CACHE_SIZE)
def post_depth(post, conv):
    """
    Returns the depth of this post within the conversation.
    Depth is defined for a node in a DAG as the longest path
    from a source node to the requested post

    Parameters
    ----------
    post : UniMessage
        The target message

    conv : Conversation
        A collection of posts

    Returns
    -------
    int
        The depth of the `post` in the conversation DAG
    """
    parent_depths = [
        post_depth(post=conv.posts[rid], conv=conv)
        for rid in post.reply_to if rid in conv.posts
    ]

    if not parent_depths:
        return 0
    else:
        return 1 + max(parent_depths)


@lru_cache(maxsize=CACHE_SIZE)
def depth_dist(conv):
    """
    Returns the depth distribution of posts within the conversation.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        The counts of posts at various depths within the conversation
    """
    return Counter([post_depth(p, conv) for p in conv.posts.values()])


@lru_cache(maxsize=CACHE_SIZE)
def post_width(post, conv):
    """
    Returns the width of the depth-level that `post` is at
    within `conv`.

    Parameters
    ----------
    post : UniMessage
        The target message

    conv : Conversation
        A collection of posts

    Returns
    -------
    int
        The width of the depth level of `posts` in `conv`
    """
    return depth_dist(conv)[post_depth(post, conv)]


def agg_post_stats(convo, filter_by=None):
    """
    Computes a set of aggregate post statistical measures.
    This is only computed for the integer and float subsets.
    Specifically, the following stats are measured:
    min, max, mean, median, standard deviation


    Parameters
    ----------
    convo : Conversation
    filter_by : function (UniMessage -> bool)

    Returns
    -------
    dict(str, dict(str, float))
    """
    return agg_post_stats_([convo], filter_by=filter_by)


def agg_post_stats_(convos, filter_by=None):
    """
    Computes a set of aggregate post statistical measures.
    This is only computed for the integer and float subsets.
    Specifically, the following stats are measured:
    min, max, mean, median, standard deviation


    Parameters
    ----------
    convos : List(Conversation)
    filter_by : function (UniMessage -> bool)

    Returns
    -------
    dict(str, dict(str, float))
    """
    agg = defaultdict(list)
    for convo in convos:
        for p in convo.posts.values():
            if filter_by is not None and not filter_by(p):
                continue

            for k, v in PostInConvoFeatures.floats(p, convo).items():
                agg[k].append(v)

            for k, v in PostInConvoFeatures.ints(p, convo).items():
                agg[k].append(v)

    out = {}
    for k, vs in agg.items():
        out[f'post_min_{k}'] = float(np.nanmin(vs))
        out[f'post_max_{k}'] = float(np.nanmax(vs))
        out[f'post_mean_{k}'] = float(np.nanmean(vs))
        out[f'post_median_{k}'] = float(np.median(vs))
        out[f'post_std_{k}'] = float(np.nanstd(vs))

    return out


def sum_booleans_across_convo(convo):
    """
    Aggregates the boolean properties of this conversation.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    dict(str, int)
    """
    cnt = Counter()
    for p in convo.posts.values():
        for k, v in PostInConvoFeatures.bools(p, convo).items():
            kx = k.replace('is_', '') + '_count'
            cnt[kx] += 1 if v else 0

    return dict(cnt)


def sum_ints_across_convo(convo):
    """
    Aggregates the boolean properties of this conversation.

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    dict(str, int)
    """
    skipset = {
        'type_count',  # must be aggregated in set theoretic way
        'depth', 'width',  # nonsensical accumulation stats
    }
    cnt = Counter()
    for p in convo.posts.values():
        for k, v in PostInConvoFeatures.ints(p, convo).items():
            if k in skipset:
                continue

            cnt[k] += v

    return dict(cnt)
