import re

from ..utils import agg_nums_in_conversation
from ..utils import memoize

URL_REGEX = re.compile(r'(\b(https?|ftp|file)://)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')


@memoize
def post_urls(post):
    """
    Given a post, extracts the URLs mentioned within its text

    Parameters
    ----------
    post : UniMessage
        The post for feature extraction

    Returns
    -------
    2-tuple(int, list(str))
        The # of URLs contained in the post and a list of those URLs
    """
    urls = [x.group() for x in URL_REGEX.finditer(post.text)]

    return len(urls), urls


def post_url_cnt(post):
    """
    Returns just the URL count

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    int
    """
    return post_urls(post=post)[0]


@memoize
def post_user_mentions(post):
    """
    Given a post, extracts the users mentioned within its text

    Parameters
    ----------
    post : UniMessage
        The post for feature extraction

    Returns
    -------
    2-tuple(int, list(str))
        The # of user mentions contained in the post and a list of those mentions
    """
    if post.MENTION_REGEX is None:
        return 0, []

    mentions = [x.group() for x in re.finditer(post.MENTION_REGEX, post.text)]

    return len(mentions), mentions


def post_mention_cnt(post):
    """
    Returns just the mention count

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    int
    """
    return post_user_mentions(post=post)[0]


@memoize
def convo_url_stats(conv):
    """
    Returns aggregated statistics about the number of URLs within the conversation.

    Parameters
    ----------
    conv : Conversation

    Returns
    -------
    dict
    """
    return agg_nums_in_conversation(conv, post_url_cnt, use_conv=False)


@memoize
def convo_mention_stats(conv):
    """
    Returns aggregated statistics about the number of mentions within the conversation.

    Parameters
    ----------
    conv : Conversation

    Returns
    -------
    dict
    """
    return agg_nums_in_conversation(conv, post_mention_cnt, use_conv=False)


__all__ = [
    'post_urls',
    'post_user_mentions',
    'convo_url_stats',
    'convo_mention_stats',
]
