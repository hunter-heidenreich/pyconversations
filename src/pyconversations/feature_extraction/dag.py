from collections import Counter

import networkx as nx

from .base import FeatureCache


class DAGFeatures(FeatureCache):

    def post_out_degree(self, post, conv=None):
        return self.wrap(post.uid, 'post_out_degree', post_out_degree, post=post, conv=conv)

    def post_in_degree(self, post, conv):
        return self.wrap(self.merge_ids(post, conv), 'post_in_degree', post_in_degree, post=post, conv=conv, cache=self)

    def convo_in_degrees(self, conv):
        return self.wrap(conv.convo_id, 'convo_in_degrees', convo_in_degrees, conv=conv)

    def convo_messages(self, conv):
        return self.wrap(conv.convo_id, 'messages', convo_messages, conv=conv)

    def convo_connections(self, conv, check=True):
        return self.wrap(conv.convo_id, 'connections', convo_connections, conv=conv, check=check)

    def convo_user_count(self, conv):
        return self.wrap(conv.convo_id, 'user_cnt', convo_user_count, conv=conv)

    def convo_messages_per_user(self, conv):
        return self.wrap(conv.convo_id, 'post_per_user', convo_messages_per_user, conv=conv)

    def convo_density(self, conv):
        return self.wrap(conv.convo_id, 'nx.density', convo_density, conv=conv)

    def convo_post_depth(self, post, conv):
        return self.wrap(self.merge_ids(post, conv), 'post_depth', convo_post_depth, post=post, conv=conv, cache=self)

    def convo_depth_dist(self, conv):
        return self.wrap(conv.convo_id, 'depth_dist', convo_depth_dist, conv=conv, cache=self)

    def convo_tree_depth(self, conv):
        return self.wrap(conv.convo_id, 'tree_depth', convo_tree_depth, conv=conv, cache=self)

    def convo_tree_width(self, conv):
        return self.wrap(conv.convo_id, 'tree_width', convo_tree_width, conv=conv, cache=self)


def post_out_degree(post, conv=None):
    """
    Returns the out-degree (# of posts replied to) of this post.
    If the conversation is specified, specifically restricts to the reply actions
    that have a valid recipient (e.g., the post replied to is in the Conversation)

    Parameters
    ----------
    post : UniMessage
        The message to compute the out-degree of

    conv : Conversation
        An optional collection of posts. If specified, replies only count if the reply is in the Conversation

    Returns
    -------
    int
        The out-degree of the `post`
    """
    return len([rid for rid in post.reply_to if rid in conv.posts]) if conv else len(post.reply_to)


def post_in_degree(post, conv, cache=None):
    """
    Returns the in-degree (# of posts that reply to this one) of this post.
    The conversation must be specified.

    Parameters
    ----------
    post : UniMessage
        The message to compute the out-degree of

    conv : Conversation
        A collection of posts

    cache : DAGFeatures
        An optional cache

    Returns
    -------
    int
        The in-degree of the `post`
    """
    cnt = cache.convo_in_degrees(conv) if cache else convo_in_degrees(conv)
    return cnt[post.uid]


def convo_in_degrees(conv):
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
        for r in p.reply_to:
            if r in conv.posts:
                cnt[r] += 1
    return cnt


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
    return len({p.author for p in conv.posts.values()})


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


def convo_post_depth(post, conv, cache=None):
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

    cache : DAGFeatures
        An optional cache

    Returns
    -------
    int
        The depth of the `post` in the conversation DAG
    """
    parent_depths = [
        cache.convo_post_depth(conv.posts[rid], conv) if cache else convo_post_depth(conv.posts[rid], conv)
        for rid in post.reply_to if rid in conv.posts
    ]

    if not parent_depths:
        return 0
    else:
        return 1 + max(parent_depths)


def convo_depth_dist(conv, cache=None):
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
    depths = [cache.convo_post_depth(p, conv) if cache else convo_post_depth(p, conv) for p in conv.posts.values()]
    return Counter(depths)


def convo_tree_depth(conv, cache=None):
    """
    Returns the depth of the full conversation.
    This is the max depth of any post within the Conversation.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    cache : DAGFeatures
        An optional cache

    Returns
    -------
    int
        Depth of the entire conversation as a DAG
    """
    dist = cache.convo_depth_dist(conv) if cache else convo_depth_dist(conv)
    return max(dist.keys())


def convo_tree_width(conv, cache=None):
    """
    Returns the width of the full conversation.
    This is the max width (# of posts) at any depth level within the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    cache : DAGFeatures
        An optional cache

    Returns
    -------
    int
        Width of the entire conversation as a DAG
    """
    dist = cache.convo_depth_dist(conv) if cache else convo_depth_dist(conv)
    return max(dist.values())
