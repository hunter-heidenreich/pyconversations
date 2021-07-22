from collections import Counter

from ..utils import memoize


@memoize
def post_char_len(post):
    """
    Given a post, returns the length of its text
    in characters.

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    int
        The number of characters in this post
    """
    return len(post.text)


@memoize
def post_tok_len(post):
    """
    Given a post, returns the length of its text
    in tokens (per its tokenizer).

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    int
        The number of tokens in this post
    """
    return len(post.tokens)


@memoize
def post_tok_dist(post):
    """
    The unigram frequency distribution of tokens within this message

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    Counter
        A counter of the types and the number of times they occur
    """
    return Counter(post.tokens)


@memoize
def post_types(post):
    """
    Returns the set of unique tokens contained within the text of the post

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    set(str)
        The set of unique token types
    """
    return set(post.tokens)


def post_type_len(post):
    """
    Given a post, returns the length of its text
    in unique token types.

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    int
        Number of unique tokens in this post
    """
    return len(post_types(post=post))
