from collections import Counter

import networkx as nx

from ..utils import memoize
from ..utils import sum_bools_in_conversation
from .post import is_post_internal_node
from .post import is_post_leaf
from .post import is_post_source
from .post import post_out_degree
from .shared import convo_in_degrees


@memoize
def convo_out_degrees(conv):
    """
    Returns a Counter of the post IDs mapping to the # of replies that post generated in this Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        A mapping from post IDs to the # of replies they direct in `conv`
    """
    return Counter({post.uid: post_out_degree(post=post, conv=conv) for post in conv.posts.values()})


@memoize
def convo_degrees(conv):
    """
    Returns a Counter of the post IDs mapping to their degrees (total replies in and out)

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        A mapping from post IDs to their degree in `conv`
    """
    return convo_in_degrees(conv=conv) + convo_out_degrees(conv=conv)


@memoize
def convo_in_degrees_dist(conv):
    """
    Returns a Counter mapping in-degree (replies received)
    to the number of posts with this size in `conv`.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        Replies received -> # of posts
    """
    return Counter(list(convo_in_degrees(conv=conv).values()))


@memoize
def convo_out_degrees_dist(conv):
    """
    Returns a Counter mapping out-degree (replies sent out, messages referenced)
    to the number of posts with this size in `conv`.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        Replies out -> # of posts
    """
    return Counter(list(convo_out_degrees(conv=conv).values()))


@memoize
def convo_degrees_dist(conv):
    """
    Returns a Counter of posts by their size (degree)

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        Size distribution of post degree
    """
    return Counter(list(convo_degrees(conv=conv).values()))


@memoize
def convo_messages(conv):
    """
    Returns the number of messages in the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    Returns
    -------
    int
        The number of messages contained in this collection
    """
    return len(conv.posts)


@memoize
def convo_sources(conv):
    """
    Returns the number of source messages in the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    Returns
    -------
    int
        The number of source messages contained in this collection
    """
    return sum_bools_in_conversation(conv, is_post_source)


@memoize
def convo_leaves(conv):
    """
    Returns the number of leaf messages (messages with no replies)
    in the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    Returns
    -------
    int
        The number of leaf messages contained in this collection
    """
    return sum_bools_in_conversation(conv, is_post_leaf)


@memoize
def convo_internal_nodes(conv):
    """
    Returns the number of internal node messages
    in the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    Returns
    -------
    int
        The number of internal node messages contained in this collection
    """
    return sum_bools_in_conversation(conv, is_post_internal_node)


@memoize
def convo_connections(conv, check=True):
    """
    Returns the number of message reply connections in the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    check : bool
        Whether to check if a reply_to ID exists in the Conversation. If True, connections without associated
        are not counted.

    Returns
    -------
    int
        The number of message reply connections contained in this collection
    """
    return len([1 for p in conv.posts.values() for ix in p.reply_to if (ix in conv.posts or not check)])


@memoize
def convo_messages_per_user(conv):
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


@memoize
def convo_user_count(conv):
    """
    Given a Conversation, returns the # of unique users who created messages within it

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    Returns
    -------
    int
        The number of unique authors (users) of messages in the Conversation
    """
    return len(convo_messages_per_user(conv=conv))


@memoize
def convo_density(conv):
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


@memoize
def convo_post_depth(post, conv):
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
        convo_post_depth(post=conv.posts[rid], conv=conv)
        for rid in post.reply_to if rid in conv.posts
    ]

    if not parent_depths:
        return 0
    else:
        return 1 + max(parent_depths)


@memoize
def convo_depth_dist(conv):
    """
    Returns the depth distribution of posts within the conversation.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    cache : DAGFeatures
        An optional cache

    Returns
    -------
    Counter
        The counts of posts at various depths within the conversation
    """
    depths = [convo_post_depth(post=p, conv=conv) for p in conv.posts.values()]
    return Counter(depths)


@memoize
def convo_tree_depth(conv):
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
    return max(convo_depth_dist(conv=conv).keys())


@memoize
def convo_post_width(post, conv):
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
    depth = convo_post_depth(post=post, conv=conv)
    dist = convo_depth_dist(conv=conv)
    return dist[depth]


@memoize
def convo_tree_width(conv):
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
    return max(convo_depth_dist(conv=conv).values())


@memoize
def convo_tree_degree(conv):
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
    return max(convo_degrees(conv=conv).values())
