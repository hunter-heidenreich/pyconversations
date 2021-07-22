from ..utils import agg_nums_in_conversation
from ..utils import memoize
from .conv import convo_duration
from .conv import convo_end_time
from .conv import convo_start_time
from .conv import convo_timeseries
from .post import post_reply_time
from .post import post_to_source


@memoize
def convo_reply_times(conv):
    """
    Aggregates statistics about the reply times of posts within this conversation

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    dict
    """
    return agg_nums_in_conversation(conv, post_reply_time)


@memoize
def convo_post_ages(conv):
    """
    Aggregates statistics about the ages of posts within this conversation (relative to the source)

    Parameters
    ----------
    convo : Conversation

    Returns
    -------
    dict
    """
    return agg_nums_in_conversation(conv, post_to_source)


__all__ = [
    'convo_duration',
    'convo_end_time',
    'convo_start_time',
    'convo_timeseries',

    'convo_post_ages',
    'convo_reply_times',

    'post_reply_time',
    'post_to_source',
]
