from ..utils import memoize
from .shared import convo_in_degrees


@memoize
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


@memoize
def post_in_degree(post, conv):
    """
    Returns the in-degree (# of posts that reply to this one) of this post.
    The conversation must be specified.

    Parameters
    ----------
    post : UniMessage
        The message to compute the out-degree of

    conv : Conversation
        A collection of posts

    Returns
    -------
    int
        The in-degree of the `post`
    """
    return convo_in_degrees(conv=conv)[post.uid]


@memoize
def post_degree(post, conv):
    """
    Returns the degree of this post.
    The conversation must be specified.

    Parameters
    ----------
    post : UniMessage
        The message to compute the out-degree of

    conv : Conversation
        A collection of posts

    Returns
    -------
    int
        The degree of the `post`
    """
    return post_in_degree(post=post, conv=conv) + post_out_degree(post=post, conv=conv)


@memoize
def is_post_source(post, conv=None):
    """
    Returns a boolean whether `post` is a source in `conv` (e.g., originator post, conversational origin)

    Parameters
    ----------
    post : UniMessage
        The target message

    conv : Conversation
        A collection of posts

    Returns
    -------
    bool
        Whether `post` is a source in `conv`
    """
    return post_out_degree(post=post, conv=conv) == 0


@memoize
def is_post_leaf(post, conv):
    """
    Returns a boolean whether `post` is a leaf (no replies for this post)
    in `conv`. (e.g., originator post, conversational origin)

    Parameters
    ----------
    post : UniMessage
        The target message

    conv : Conversation
        A collection of posts

    Returns
    -------
    bool
        Whether `post` is a leaf in `conv`
    """
    return post_in_degree(post=post, conv=conv) == 0


@memoize
def is_post_internal_node(post, conv):
    """
    Returns a boolean whether `post` is a leaf (no replies for this post)
    in `conv`. (e.g., originator post, conversational origin)

    Parameters
    ----------
    post : UniMessage
        The target message

    conv : Conversation
        A collection of posts

    Returns
    -------
    bool
        Whether `post` is an internal node in `conv`
    """
    return not is_post_source(post=post, conv=conv) and not is_post_leaf(post=post, conv=conv)


@memoize
def is_post_source_author(post, conv):
    """
    Returns a boolean indicating whether this post is written by an author that is also a source post author
    """
    sids = conv.get_sources()
    auths = set([conv.posts[s].author for s in sids])
    return post.author in auths
