from collections import Counter
from functools import lru_cache
from functools import reduce

from ..convo import Conversation
from .harmonic import mixing
from .harmonic import novelty
from .params import CACHE_SIZE
from .post import type_frequency_distribution as post_freq
from .post_in_conv import agg_post_stats
from .post_in_conv import sum_booleans_across_convo as sum_post_bools
from .post_in_conv import sum_ints_across_convo as sum_post_ints
from .utils import apply_extraction


def collapse_convos(convos):
    """
    Given a list of conversations,
    collpases them into one mega container.

    Parameters
    ----------
    convos : List(Conversation)

    Returns
    -------
    Conversation
    """
    all_posts = Conversation()
    for convo in convos:
        all_posts += convo

    return all_posts


def iter_over_users(convo):
    """
    Creates an iterator over the unique users within a conversation

    Parameters
    ----------
    convo : Conversation

    Yields
    -------
    str
        A user
    """
    users = set()
    for pid in convo.posts:
        user = convo.posts[pid].author

        if user in users:
            continue

        yield user

        users.add(user)


def get_all(ux, cx, keys=None, ignore_keys=None, include_post=True):
    """
    Returns all features for a user in a conversation

    Parameters
    ----------
    ux : str
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)
    include_post : bool

    Returns
    -------
    dict(str, Any)
    """
    out = {
        **get_bools(ux, cx, keys, ignore_keys),
        **get_floats(ux, cx, keys, ignore_keys, include_post),
        **get_ints(ux, cx, keys, ignore_keys),
    }

    return out


def get_bools(ux, cx, keys=None, ignore_keys=None):
    """
    Returns the boolean features for a user in a conversation.

    Parameters
    ----------
    ux : str
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, bool)
    """
    return apply_extraction({
        'is_source_author': is_source_author,
    }, keyset=keys, ignore=ignore_keys, user=ux, convo=cx)


def get_floats(ux, cx, keys=None, ignore_keys=None, include_post=True):
    """
    Returns the float features for a user in a conversation.

    Parameters
    ----------
    ux : str
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)
    include_post : bool

    Returns
    -------
    dict(str, float)
    """
    out = {
        **apply_extraction({
            'mixing_k1':      lambda user, convo: mixing_features(user, convo)['k1'],
            'mixing_theta':   lambda user, convo: mixing_features(user, convo)['theta'],
            'mixing_entropy': lambda user, convo: mixing_features(user, convo)['entropy'],
            'mixing_N_avg':   lambda user, convo: mixing_features(user, convo)['N_avg'],
            'mixing_M_avg':   lambda user, convo: mixing_features(user, convo)['M_avg'],
        }, keyset=keys, ignore=ignore_keys, convo=cx, user=ux)
    }

    if include_post:
        out = {**out, **agg_post_stats(cx, keys=keys, ignore={
            'mixing_k1', 'mixing_theta', 'mixing_entropy',
            'mixing_N_avg', 'mixing_M_avg',
        } | set([] if ignore_keys is None else ignore_keys), filter_by=lambda p: p.author == ux)}

    return out


def get_ints(ux, cx, keys=None, ignore_keys=None):
    """
    Returns all integer features for a user in a conversation

    Parameters
    ----------
    ux : str
    cx : Conversation
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, int)
    """
    return {
        **apply_extraction({
            'message_count': messages_by_user,
            'types':         lambda user, convo: len(type_frequency_distribution(user, convo)),
        }, keyset=keys, ignore=ignore_keys, convo=cx, user=ux),
        **sum_post_bools(get_user_posts(ux, cx)),
        **sum_post_ints(get_user_posts(ux, cx)),
    }


@lru_cache(maxsize=CACHE_SIZE)
def is_source_author(user, convo):
    """
    Returns if this user created a source message
    for the conversation

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    bool
    """
    auths = {convo.posts[pid].author for pid in convo.get_sources()}
    return user in auths


@lru_cache(maxsize=CACHE_SIZE)
def messages_per_user(conv):
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


@lru_cache(maxsize=CACHE_SIZE)
def messages_by_user(user, conv):
    """
    Returns number of messages created by this user
    Parameters
    ----------
    user : str
    conv : Conversation

    Returns
    -------
    int
    """
    return messages_per_user(conv)[user]


@lru_cache(maxsize=CACHE_SIZE)
def get_user_posts(user, conv):
    """
    Filters to just this users messages within the conversation

    Parameters
    ----------
    user : str
    conv : Conversation

    Returns
    -------
    Conversation
    """
    return conv.filter(by_author=user)


@lru_cache(maxsize=CACHE_SIZE)
def type_frequency_distribution(user, convo):
    """
    Returns the type frequency (unigram) distribution for the user's posts in the convo.

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    collections.Counter
    """
    return reduce(lambda x, y: x + y, map(post_freq, get_user_posts(user, convo).posts.values()))


@lru_cache(maxsize=CACHE_SIZE)
def mixing_features(user, convo):
    """
    Returns the measured parameters using the harmonic mixing law.

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    dict(str, float)
    """
    return mixing(type_frequency_distribution(user, convo))


@lru_cache(maxsize=CACHE_SIZE)
def novelty_vector(user, convo):
    """
    Returns the novelty vector measured from the convo text, filtered to just the user.

    Parameters
    ----------
    user : str
    convo : Conversation

    Returns
    -------
    np.array
    """
    return novelty(type_frequency_distribution(user, convo))
