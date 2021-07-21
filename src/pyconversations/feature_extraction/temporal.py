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

    def post_reply_time(self, post, conv):
        return self.wrap(self.merge_ids(post, conv), 'reply_time', post_reply_time, post=post, conv=conv)

    def post_to_source(self, post, conv):
        return self.wrap(self.merge_ids(post, conv), 'to_source', post_to_source, post=post, conv=conv)


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


def post_reply_time(post, conv):
    """
    Returns the time between the post and its parent

    Parameters
    ----------
    post : UniMessage
        The message

    conv : Conversation
        A collection of posts

    Returns
    -------
    float
       The time between the `post` and its parent. If multiple parents, returns the minimum response difference
    """
    diffs = [
        (post.created_at - conv.posts[rid].created_at).total_seconds()
        for rid in post.reply_to if rid in conv.posts
    ]

    if not diffs:
        return 0

    return min(diffs)


def post_to_source(post, conv):
    """
    Returns the time between the post and the conversation source

    Parameters
    ----------
    post : UniMessage
        The message

    conv : Conversation
        A collection of posts

    Returns
    -------
    float
       The time between the `post` and its parent. If multiple sources, returns the maximum response difference
    """
    timeorder = conv.time_order()

    if not timeorder:
        return 0

    return post.created_at.timestamp() - timeorder[0]
