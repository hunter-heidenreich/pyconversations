from collections import Counter
from collections import defaultdict

import numpy as np

from ..convo import Conversation
from .user_in_conv import get_bools as uic_bools
from .user_in_conv import get_floats as uic_floats
from .user_in_conv import get_ints as uic_ints
from .user_in_conv import mixing_features
from .utils import apply_extraction


def get_all(ux, cxs, keys=None, ignore_keys=None):
    """
    Returns all features for a user across multiple conversations

    Parameters
    ----------
    ux : str
    cxs : List(Conversation)
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, Any)
    """
    out = {
        **get_floats(ux, cxs, keys, ignore_keys),
        **get_ints(ux, cxs, keys, ignore_keys),
    }

    return out


def get_floats(ux, cxs, keys=None, ignore_keys=None):
    """
    Returns the float features for a user across multiple conversations

    Parameters
    ----------
    ux : str
    cxs : List(Conversation)
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, float)
    """
    user_posts = Conversation(convo_id=ux)
    for cx in cxs:
        if ux not in cx.authors:
            continue

        for pid in cx.filter(by_author=ux):
            user_posts.add_post(cx.posts[pid])

    out = {
        **apply_extraction({
            'mixing_k1': lambda user, convo: mixing_features(user, convo)['k1'],
            'mixing_theta': lambda user, convo: mixing_features(user, convo)['theta'],
            'mixing_entropy': lambda user, convo: mixing_features(user, convo)['entropy'],
            'mixing_N_avg': lambda user, convo: mixing_features(user, convo)['N_avg'],
            'mixing_M_avg': lambda user, convo: mixing_features(user, convo)['M_avg'],
        }, keyset=keys, ignore=ignore_keys, convo=user_posts, user=ux),
        **agg_user_stats_across(ux, cxs, keys, ignore_keys),
    }

    return out


def get_ints(ux, cxs, keys=None, ignore_keys=None):
    """
    Returns all integer features for a user across multiple conversations

    Parameters
    ----------
    ux : str
    cxs : List(Conversation)
    keys : None or Iterable(str)
    ignore_keys : None or Iterable(str)

    Returns
    -------
    dict(str, int)
    """
    return {
        **sum_user_booleans_across_convos(ux, cxs, keys, ignore_keys),
        **sum_user_ints_across_convos(ux, cxs, keys, ignore_keys),
    }


def agg_user_stats_across(user, convos, keys=None, ignore=None):
    """
    Computes a set of aggregate user statistical measures.
    This is only computed for the integer and float subsets.
    Specifically, the following stats are measured:
    min, max, mean, median, standard deviation


    Parameters
    ----------
    user : str
    convos : List(Conversation)
    keys : None or Iterable(str)
    ignore : None or Iterable(str)

    Returns
    -------
    dict(str, dict(str, float))
    """
    agg = defaultdict(list)
    fs = [uic_floats, uic_ints]
    for convo in convos:
        if user not in convo.authors:
            continue

        for f in fs:
            for k, v in f(user, convo, keys=keys, ignore_keys=ignore).items():
                agg[k].append(v)

    out = {}
    for k, vs in agg.items():
        out[f'user_min_{k}'] = float(np.nanmin(vs))
        out[f'user_max_{k}'] = float(np.nanmax(vs))
        out[f'user_mean_{k}'] = float(np.nanmean(vs))
        out[f'user_median_{k}'] = float(np.median(vs))
        out[f'user_std_{k}'] = float(np.nanstd(vs) if len(vs) > 1 else 1)

    return out


def sum_user_booleans_across_convos(user, convos, keys=None, ignore=None):
    """
    Aggregates the boolean properties of this user across conversations.

    Parameters
    ----------
    user : str
    convos : list(Conversation)
    keys : None or Iterable(str)
    ignore : None or Iterable(str)

    Returns
    -------
    dict(str, int)
    """
    cnt = Counter()
    for convo in convos:
        if user not in convo.authors:
            continue

        for k, v in uic_bools(user, convo, keys=keys, ignore_keys=ignore).items():
            kx = k.replace('is_', '') + '_count'
            cnt[kx] += 1 if v else 0

    return dict(cnt)


def sum_user_ints_across_convos(user, convos, keys=None, ignore=None):
    """
    Aggregates the integer properties of this user across conversations.

    Parameters
    ----------
    user : str
    convos : list(Conversation)
    keys : None or Iterable(str)
    ignore : None or Iterable(str)

    Returns
    -------
    dict(str, int)
    """
    skipset = {
        'type_count',  # must be aggregated in set theoretic way
    }
    cnt = Counter()
    for convo in convos:
        if user not in convo.authors:
            continue

        for k, v in uic_bools(user, convo, keys=keys, ignore_keys=ignore).items():
            if k in skipset:
                continue

            cnt[k] += v

    return dict(cnt)
