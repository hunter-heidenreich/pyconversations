from collections import Counter
from functools import lru_cache

from demoji import findall_list

from .harmonic import mixing
from .harmonic import novelty
from .params import CACHE_SIZE
from .regex import HASHTAG_REGEX
from .regex import URL_REGEX
from .regex import get_all as get_all_regex
from .utils import apply_extraction


def get_all(px, keys=None, ignore_keys=None):
    """
    Returns all features specified in keys or all features minus what is specified in ignore_keys.

    Parameters
    ----------
    px : UniMessage
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, Any)
    """
    return {
        **get_bools(px, keys, ignore_keys),
        **get_categorical(px, keys, ignore_keys),
        **get_counters(px, keys, ignore_keys),
        **get_floats(px, keys, ignore_keys),
        **get_ints(px, keys, ignore_keys),
        **get_substrings(px, keys, ignore_keys),
    }


def get_bools(px, keys=None, ignore_keys=None):
    """
    Returns the boolean features.
    In keys, one may specify a subset of features to extract.
    Alternatively, one can specify a set of keys to ignore.

    Parameters
    ----------
    px : UniMessage
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, bool)
    """
    return apply_extraction({
        'is_source': is_source,
    }, keyset=keys, ignore=ignore_keys, post=px)


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


def get_categorical(px, keys=None, ignore_keys=None):
    """
    Returns the categorical string features.
    In keys, one may specify a subset of features to extract.
    Alternatively, one can specify a set of keys to ignore.

    Parameters
    ----------
    px : UniMessage
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, str)
    """
    return apply_extraction({
        'author': lambda post: post.author,
        'language': lambda post: post.lang,
        'platform': lambda post: post.platform,
    }, keyset=keys, ignore=ignore_keys, post=px)


def get_floats(px, keys=None, ignore_keys=None):
    """
    Returns the floating point features measured directly from the post.
    By specifying keys with the optional `keys` parameter,
    a subset of the features may be returned.
    Alternatively, one can specify a set of keys to ignore.

    Parameters
    ----------
    px : UniMessage
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, float)
    """
    return apply_extraction({
        'mixing_k1': lambda post: mixing_features(post)['k1'],
        'mixing_theta': lambda post: mixing_features(post)['theta'],
        'mixing_entropy': lambda post: mixing_features(post)['entropy'],
        'mixing_N_avg': lambda post: mixing_features(post)['N_avg'],
        'mixing_M_avg': lambda post: mixing_features(post)['M_avg'],
    }, keyset=keys, ignore=ignore_keys, post=px)


def get_ints(px, keys=None, ignore_keys=None):
    """
    Given a post, returns all supported features that are integers.
    By specifying keys with the optional `keys` parameter,
    a subset of the features may be returned.
    Alternatively, one can specify a set of keys to ignore.

    Parameters
    ----------
    px : UniMessage
    keys : Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, int)
        A dictionary of integer stats
    """
    return apply_extraction({
        '?_count': lambda post: len(get_all_regex(post, r'[?]')),
        '!_count': lambda post: len(get_all_regex(post, r'[!]')),
        'char_count': lambda post: len(post.text),
        'emoji_count': lambda post: len(emojis(post)),
        'hashtag_count': lambda post: len(hashtags(post)),
        'mention_count': lambda post: len(mentions(post)),
        'degree_out':      out_degree,
        'punct_count': lambda post: len(get_all_regex(post, r'[,.?!;\'"]')),
        'token_count': lambda post: len(post.tokens),
        'type_count': lambda post: len(type_frequency_distribution(post)),
        'uppercase_count': lambda post: len(get_all_regex(post, r'[A-Z]')),
        'url_count': lambda post: len(urls(post)),
    }, keyset=keys, ignore=ignore_keys, post=px)


def get_substrings(px, keys=None, ignore_keys=None):
    """
    Given a post, returns all supported features that are lists of substrings.
    By specifying keys with the optional `keys` parameter,
    a subset of the features may be returned.
    Alternatively, one can specify a set of keys to ignore.

    Parameters
    ----------
    px : UniMessage
    keys : Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, list(str))
    """
    return apply_extraction({
        'emojis':   emojis,
        'hashtags': hashtags,
        'mentions': mentions,
        'tokens': lambda post: post.tokens,
        'urls':     urls,
    }, keyset=keys, ignore=ignore_keys, post=px)


def get_counters(px, keys=None, ignore_keys=None):
    """
    Returns a set of Counter objects generated from this post.
    By specifying keys with the optional `keys` parameter,
    a subset of the features may be returned.
    Alternatively, one can specify a set of keys to ignore.

    Parameters
    ----------
    px : UniMessage
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, Counter)
    """
    return apply_extraction({
        'type_frequency': type_frequency_distribution,
    }, keyset=keys, ignore=ignore_keys, post=px)


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
