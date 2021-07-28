from collections import Counter
from collections import defaultdict
from functools import lru_cache
from functools import reduce

import numpy as np

from ..convo import Conversation
from . import CACHE_SIZE
from .post import get_all as post_get_all
from .post import get_bools as post_get_bools
from .post import get_floats as post_get_floats
from .post import get_ints as post_get_ints
from .utils import apply_extraction


def get_all(px, cx, keys=None, ignore_keys=None, include_static=True):
    """
    Returns all features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    px : UniMessage
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)
    include_static : bool
        Default True

    Returns
    -------
    dict(str, Any)
    """
    out = {
        **get_bools(px, cx, keys, ignore_keys),
        **get_floats(px, cx, keys, ignore_keys),
        **get_ints(px, cx, keys, ignore_keys),
    }

    if include_static:
        out = {**out, **post_get_all(px, keys, ignore_keys)}

    return out


def get_bools(px, cx, keys=None, ignore_keys=None, include_static=True):
    """
    Returns all boolean features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    px : UniMessage
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)
    include_static : bool
        Default True

    Returns
    -------
    dict(str, bool)
    """
    out = apply_extraction({
        'is_leaf':                 lambda post, convo: in_degrees_by_uid(convo)[post.uid] == 0,
        'is_internal':             lambda post, convo: in_degrees_by_uid(convo)[post.uid] != 0 and not
        post_get_all(post, keys={'is_source'})['is_source'],
        'is_author_source_author': lambda post, conv: post.author in source_authors(conv),
    }, keyset=keys, ignore=ignore_keys, post=px, convo=cx)

    if include_static:
        out = {**out, **post_get_bools(px, keys, ignore_keys)}

    return out


def get_floats(px, cx, keys=None, ignore_keys=None, include_static=True):
    """
    Returns all float features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    px : UniMessage
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)
    include_static : bool
        Default True

    Returns
    -------
    dict(str, float)
    """
    out = apply_extraction({
        'relative_age':  lambda post, convo: post_to_source(post, convo),
        'response_time': lambda post, convo: post_reply_time(post, convo),
    }, keyset=keys, ignore=ignore_keys, post=px, convo=cx)

    kx = 'avg_token_entropy'
    if kx in keys if keys is not None else (kx not in ignore_keys if ignore_keys is not None else True):
        out = {**out, **avg_token_entropy_all_splits(px, cx)}

    if include_static:
        out = {**out, **post_get_floats(px, keys, ignore_keys)}

    return out


def get_ints(px, cx, keys=None, ignore_keys=None, include_static=True):
    """
    Returns all integer features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    px : UniMessage
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)
    include_static : bool
        Default True

    Returns
    -------
    dict(str, int)
    """
    out = apply_extraction({
        'degree':               lambda post, convo: in_degrees_by_uid(convo)[post.uid] +
                                                    post_get_all(post, keys={'degree_out'})[
                                                        'degree_out'],
        'degree_in':            lambda post, convo: in_degrees_by_uid(convo)[post.uid],
        'depth':                post_depth,
        'width':                post_width,
    }, keyset=keys, ignore=ignore_keys, post=px, convo=cx)

    if include_static:
        out = {**out, **post_get_ints(px, keys, ignore_keys)}

    return out


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
    diffs = [
        (post.created_at - conv.posts[rid].created_at).total_seconds()
        for rid in post.reply_to if rid in conv.posts
    ]

    if not diffs:
        return 0.0

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
        return 0.0

    return (post.created_at - conv.posts[timeorder[0]].created_at).total_seconds()


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

    post_dist = post_get_all(post, keys={'type_frequency'})['type_frequency']
    convo_dist = reduce(lambda x, y: x + y,
                        [post_get_all(p, keys={'type_frequency'})['type_frequency'] for p in conv.posts.values()])

    conv_n = len(convo_dist)
    conv_m = sum(convo_dist.values())
    post_m = sum(post_dist.values())

    if not post_m or not conv_m:
        return 0

    entropy = -(np.log([convo_dist[w] / conv_m for w in post_dist]) / (np.log(conv_n) * post_m)).sum()
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
    if not len(conv_a.posts) or not len(conv_b.posts):
        return 0

    joint_conv = conv_a + conv_b
    joint_dist = reduce(lambda x, y: x + y,
                        [post_get_all(p, keys={'type_frequency'})['type_frequency'] for p in joint_conv.posts.values()])

    joint_n = len(joint_dist)
    joint_m = sum(joint_dist.values())

    if not joint_m or not joint_n:
        return 0

    left_dist = reduce(lambda x, y: x + y,
                        [post_get_all(p, keys={'type_frequency'})['type_frequency'] for p in conv_a.posts.values()])
    left_m = sum(left_dist.values())

    if not left_m:
        return 0

    entropy = -(np.log([joint_dist[w] / joint_m for w in left_dist]) / (np.log(joint_n) * left_m)).sum()
    return entropy


@lru_cache(maxsize=CACHE_SIZE)
def avg_token_entropy_all_splits(post, conv):
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
    splits = {
        'post':        post,
        'full':        conv,

        'ancestors':   conv.get_ancestors(post.uid, include_post=True),
        'after':       conv.get_after(post.uid, include_post=True),
        'before':      conv.get_before(post.uid, include_post=True),
        'children':    conv.get_children(post.uid, include_post=True),
        'descendants': conv.get_descendants(post.uid, include_post=True),
        'parents':     conv.get_parents(post.uid, include_post=True),
        'siblings':    conv.get_siblings(post.uid, include_post=True),
    }
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


def agg_post_stats(convo, keys=None, ignore=None, filter_by=None):
    """
    Computes a set of aggregate post statistical measures.
    This is only computed for the integer and float subsets.
    Specifically, the following stats are measured:
    min, max, mean, median, standard deviation


    Parameters
    ----------
    convo : Conversation
    keys : None or Iterable(str)
    ignore : None or Iterable(str)
    filter_by : function (UniMessage -> bool)

    Returns
    -------
    dict(str, dict(str, float))
    """
    agg = defaultdict(list)
    for p in convo.posts.values():
        if filter_by is not None and not filter_by(p):
            continue

        for k, v in get_floats(p, convo, keys=keys, ignore_keys=ignore).items():
            agg[k].append(v)

        for k, v in get_ints(p, convo, keys=keys, ignore_keys=ignore).items():
            agg[k].append(v)

    return {
        k: {
            'min': np.min(vs),
            'max': np.max(vs),
            'std': np.std(vs),
            'mean': np.mean(vs),
            'median': np.median(vs),
        }
        for k, vs in agg.items()
    }


def sum_booleans_across_convo(convo, keys=None, ignore=None):
    """
    Aggregates the boolean properties of this conversation.

    Parameters
    ----------
    convo : Conversation
    keys : None or Iterable(str)
    ignore : None or Iterable(str)

    Returns
    -------
    dict(str, int)
    """
    cnt = Counter()
    for p in convo.posts.values():
        for k, v in get_bools(p, convo, keys=keys, ignore_keys=ignore).items():
            kx = k.replace('is_', '') + '_count'
            cnt[kx] += 1 if v else 0

    return dict(cnt)


def sum_ints_across_convo(convo, keys=None, ignore=None):
    """
    Aggregates the boolean properties of this conversation.

    Parameters
    ----------
    convo : Conversation
    keys : None or Iterable(str)
    ignore : None or Iterable(str)

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
        for k, v in get_ints(p, convo, keys=keys, ignore_keys=ignore).items():
            if k in skipset:
                continue

            cnt[k] += v

    return dict(cnt)



