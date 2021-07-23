from ..utils import agg_nums_in_conversation
from ..utils import memoize
from .post import post_mention_cnt
from .post import post_url_cnt


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
