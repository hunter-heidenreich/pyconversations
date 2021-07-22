from ..utils import memoize


@memoize
def convo_start_time(conv):
    """
    Given a conversation, returns the earliest timestamp (if available)

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    float
        The earliest time stamp. 0 if time order is not available
    """
    order = conv.time_order()
    return conv.posts[order[0]].created_at.timestamp() if order else 0


@memoize
def convo_end_time(conv):
    """
    Given a conversation, returns the latest timestamp (if available)

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    float
        The latest time stamp. 0 if time order is not available
    """
    order = conv.time_order()
    return conv.posts[order[-1]].created_at.timestamp() if order else 0


@memoize
def convo_duration(conv):
    """
    Given a conversation, returns the duration (in seconds)

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    float
        The length of the conversation in seconds
    """
    start = convo_start_time(conv=conv)
    end = convo_end_time(conv=conv)
    return end - start


@memoize
def convo_timeseries(conv):
    """
    Given a conversation, returns the list of the timestamps when posts were generated

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    list(float)
        The in-order list of message creation timestamps
    """
    order = conv.time_order()
    return [conv.posts[uid].created_at.timestamp() for uid in order] if order else []
