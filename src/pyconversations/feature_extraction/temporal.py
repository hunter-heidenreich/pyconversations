from .base import FeatureCache


class TemporalFeatures(FeatureCache):

    def convo_start_time(self, conv):
        return self.wrap(conv.convo_id, 'start_time', convo_start_time, conv=conv)

    def convo_end_time(self, conv):
        return self.wrap(conv.convo_id, 'end_time', convo_end_time, conv=conv)

    def convo_duration(self, conv):
        return self.wrap(conv.convo_id, 'duration', convo_duration, conv=conv, cache=self)

    def convo_timeseries(self, conv):
        return self.wrap(conv.convo_id, 'timeseries', convo_timeseries, conv=conv)


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
        The earliest time stamp. -1 if time order is not available
    """
    order = conv.time_order()
    return conv.posts[order[0]].created_at.timestamp() if order else -1


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
        The latest time stamp. -1 if time order is not available
    """
    order = conv.time_order()
    return conv.posts[order[-1]].created_at.timestamp() if order else -1


def convo_duration(conv, cache=None):
    """
    Given a conversation, returns the duration (in seconds)

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    cache : TemporalFeatures
        An optional cache to use

    Returns
    -------
    float
        The length of the conversation in seconds
    """
    start = cache.convo_start_time(conv) if cache else convo_start_time(conv)
    end = cache.convo_end_time(conv) if cache else convo_end_time(conv)
    return end - start


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
