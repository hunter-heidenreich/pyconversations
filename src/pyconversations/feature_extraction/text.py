from collections import Counter


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
    return len(post._toks())


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
    return Counter(post._toks())


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
    return len(post._types())
