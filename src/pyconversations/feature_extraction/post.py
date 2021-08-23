from collections import Counter
from functools import lru_cache

from demoji import findall_list

from .harmonic import mixing
from .harmonic import novelty
from .params import CACHE_SIZE
from .regex import HASHTAG_REGEX
from .regex import URL_REGEX
from .regex import get_all as get_all_regex


class PostFeatures:

    """
    Container of feature extraction for posts in isolation
    """

    @staticmethod
    def bools(post):
        return {
            'is_source': is_source(post),
        }

    @staticmethod
    def categoricals(post):
        return {
            'author':   post.author,
            'lang':     post.lang,
            'platform': post.platform,
        }

    @staticmethod
    def counter(post):
        return {
            'type_freq': type_frequency_distribution(post),
        }

    @staticmethod
    def floats(post):
        return mixing_features(post)

    @staticmethod
    def ints(post):
        return {
            '?_count':         len(get_all_regex(post, r'[?]')),
            '!_count':         len(get_all_regex(post, r'[!]')),
            'char_count':      len(post.text),
            'emoji_count':     len(emojis(post)),
            'hashtag_count':   len(hashtags(post)),
            'mention_count':   len(mentions(post)),
            'out_degree':      out_degree(post),
            'punct_count':     len(get_all_regex(post, r'[,.?!;\'"]')),
            'token_count':     len(post.tokens),
            'type_count':      len(type_frequency_distribution(post)),
            'uppercase_count': len(get_all_regex(post, r'[A-Z]')),
            'url_count':       len(urls(post)),
        }

    @staticmethod
    def strs(post):
        return {
            'emojis':   emojis(post),
            'hashtags': hashtags(post),
            'mentions': mentions(post),
            'tokens':   post.tokens,
            'urls':     urls(post),
        }


def is_source(post):
    """
    Returns a bool to indicate if this post is a source (or a reply)

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    bool
    """
    return out_degree(post) == 0


def out_degree(post):
    """
    Returns the out degree of a post.

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    int
        The number of posts this post replies to, as indicated by the post object.
    """
    return len(post.reply_to)


@lru_cache(maxsize=CACHE_SIZE)
def mentions(post):
    """
    Returns the user mentions within the post

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    list(str)
    """
    return [] if post.MENTION_REGEX is None else get_all_regex(post, post.MENTION_REGEX)


@lru_cache(maxsize=CACHE_SIZE)
def urls(post):
    """
    Returns the URLs within this post

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    list(str)
    """
    return get_all_regex(post, URL_REGEX)


@lru_cache(maxsize=CACHE_SIZE)
def hashtags(post):
    """
    Returns the strings of hashtags mentioned in this post

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    list(str)
    """
    return get_all_regex(post, HASHTAG_REGEX)


@lru_cache(maxsize=CACHE_SIZE)
def emojis(post):
    """
    Returns a list of all extracted emojis.

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    list(str)
        The extracted emojis
    """
    return findall_list(post.text, desc=False)


@lru_cache(maxsize=CACHE_SIZE)
def type_frequency_distribution(post):
    """
    Returns the type frequency (unigram) distribution for the post.

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    collections.Counter
    """
    return Counter(post.tokens)


@lru_cache(maxsize=CACHE_SIZE)
def mixing_features(post):
    """
    Returns the measured parameters using the harmonic mixing law.

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    dict(str, float)
    """
    if not post.text:
        return {
            'k1':      float(0),
            'theta':   float(0),
            'entropy': float(0),
            'N_avg':   float(0),
            'M_avg':   float(0),
        }

    return mixing(type_frequency_distribution(post))


@lru_cache(maxsize=CACHE_SIZE)
def novelty_vector(post):
    """
    Returns the novelty vector measured from the post text.

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    np.array
    """
    return novelty(type_frequency_distribution(post))
