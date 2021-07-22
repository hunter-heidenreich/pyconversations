from collections import Counter
from functools import reduce

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
