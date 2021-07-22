from collections import Counter
from functools import reduce

import numpy as np

from ..utils import memoize
from .post import post_char_len
from .post import post_tok_dist
from .post import post_tok_len
from .post import post_type_len


@memoize
def convo_chars(conv):
    """
    Returns the total length of a conversation/collection of posts
    and the size distribution of posts based on the number of characters.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    Returns
    -------
    2-tuple(int, Counter)
        The total aggregate size of the conversation (in chars) as well as the distribution per post
    """
    cnt = convo_chars_per_post(conv=conv)
    return sum(map(lambda kv: kv[0] * kv[1], cnt.items()))


@memoize
def convo_chars_per_post(conv):
    """
    Returns the size distribution of posts based on the number of chars.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    Returns
    -------
    Counter
        The per-post char distribution within this conversation
    """
    return Counter([post_char_len(post=p) for p in conv.posts.values()])


@memoize
def convo_tokens(conv):
    """
    Returns the total length of a conversation/collection of posts
    and the size distribution of posts based on the number of tokens.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    Returns
    -------
    2-tuple(int, Counter)
        The total aggregate size of the conversation (in tokens) as well as the distribution per post
    """
    cnt = convo_tokens_per_post(conv=conv)
    return sum(map(lambda kv: kv[0] * kv[1], cnt.items()))


@memoize
def convo_token_dist(conv):
    """
    Returns the token frequency distribution as a Counter for the entire conversation

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    Returns
    -------
    Counter
        The token frequency distribution for the entire conversation
    """
    return reduce(lambda x, y: x + y, [post_tok_dist(post=p) for p in conv.posts.values()])


@memoize
def convo_tokens_per_post(conv):
    """
    Returns the size distribution of posts based on the number of tokens.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    Returns
    -------
    Counter
        The per-post token distribution within this conversation
    """
    return Counter([post_tok_len(post=p) for p in conv.posts.values()])


@memoize
def convo_types(conv):
    """
    Returns the number of unique token types in this conversation.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    Returns
    -------
    int
        Number of unique token types in this Conversation
    """
    return len(convo_token_dist(conv=conv))


@memoize
def convo_types_per_post(conv):
    """
    Returns the size distribution of posts based on the number of unique tokens (types).

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    Returns
    -------
    Counter
        The per-post type distribution within this conversation
    """
    return Counter([post_type_len(post=p) for p in conv.posts.values()])


@memoize
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
    conv_n = convo_types(conv=conv)
    conv_m = convo_tokens(conv=conv)
    post_m = post_tok_len(post=post)

    post_dist = post_tok_dist(post=post)
    convo_dist = convo_token_dist(conv=conv)

    entropy = -(np.log([convo_dist[w] / conv_m for w in post_dist]) / (np.log(conv_n) * post_m)).sum()
    return entropy


@memoize
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
    joint_conv = conv_a + conv_b

    joint_n = convo_types(conv=joint_conv)
    joint_m = convo_tokens(conv=joint_conv)
    joint_dist = convo_token_dist(conv=joint_conv)

    left_m = convo_tokens(conv=conv_a)
    left_dist = convo_token_dist(conv=conv_a)

    entropy = -(np.log([joint_dist[w] / joint_m for w in left_dist]) / (np.log(joint_n) * left_m)).sum()
    return entropy


@memoize
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

            entropy[key] = e

    return entropy
