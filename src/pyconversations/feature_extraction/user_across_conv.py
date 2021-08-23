from collections import Counter
from collections import defaultdict

import numpy as np

from ..convo import Conversation
from .post_in_conv import agg_post_stats_
from .user_in_conv import UserInConvoFeatures
from .user_in_conv import get_user_posts
from .user_in_conv import mixing_features
from .user_in_conv import type_frequency_distribution


class UserAcrossConvoFeatures:

    """
    Container for feature extraction on users situated within multiple conversations.
    """

    @staticmethod
    def bools(user, convos):
        return {}

    @staticmethod
    def categoricals(user, convos):
        return {}

    @staticmethod
    def counter(user, convos):
        return {
            'type_freq': type_frequency_distribution(user, get_user_posts(user, convos)),
        }

    @staticmethod
    def floats(user, convos):
        user_conv = gather_all_user_posts_in_convo(user, convos)
        out = mixing_features(user, user_conv)

        for k, v in agg_post_stats_(convos, filter_by=lambda p: p.author == user).items():
            out[k] = v

        return out

    @staticmethod
    def ints(user, convos):
        out = sum_user_ints_across_convos(user, convos)

        for k, v in sum_user_booleans_across_convos(user, convos).items():
            out[k] = v

        return out

    @staticmethod
    def strs(user, convos):
        return {}


def gather_all_user_posts_in_convo(user, convos):
    if user is None:
        return Conversation()

    x = Conversation(convo_id='all_posts: ' + user)
    for conv in convos:
        if user not in conv.authors:
            continue

        for pid in conv.filter(by_author=user):
            x.add_post(conv.posts[pid])

    return x


def agg_user_stats_across(user, convos):
    """
    Computes a set of aggregate user statistical measures.
    This is only computed for the integer and float subsets.
    Specifically, the following stats are measured:
    min, max, mean, median, standard deviation


    Parameters
    ----------
    user : str
    convos : List(Conversation)

    Returns
    -------
    dict(str, dict(str, float))
    """
    agg = defaultdict(list)
    fs = [UserInConvoFeatures.floats, UserInConvoFeatures.ints]
    for convo in convos:
        if user not in convo.authors:
            continue

        for f in fs:
            for k, v in f(user, convo).items():
                agg[k].append(v)

    out = {}
    for k, vs in agg.items():
        out[f'user_min_{k}'] = float(np.nanmin(vs))
        out[f'user_max_{k}'] = float(np.nanmax(vs))
        out[f'user_mean_{k}'] = float(np.nanmean(vs))
        out[f'user_median_{k}'] = float(np.median(vs))
        out[f'user_std_{k}'] = float(np.nanstd(vs) if len(vs) > 1 else 1)

    return out


def sum_user_booleans_across_convos(user, convos):
    """
    Aggregates the boolean properties of this user across conversations.

    Parameters
    ----------
    user : str
    convos : list(Conversation)

    Returns
    -------
    dict(str, int)
    """
    cnt = Counter()
    for convo in convos:
        if user not in convo.authors:
            continue

        for k, v in UserInConvoFeatures.bools(user, convo).items():
            kx = k.replace('is_', '') + '_count'
            cnt[kx] += 1 if v else 0

    return dict(cnt)


def sum_user_ints_across_convos(user, convos):
    """
    Aggregates the integer properties of this user across conversations.

    Parameters
    ----------
    user : str
    convos : list(Conversation)

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

        for k, v in UserInConvoFeatures.ints(user, convo).items():
            if k in skipset:
                continue

            cnt[k] += v

    return dict(cnt)
