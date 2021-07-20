from collections import Counter

import networkx as nx

from .base import FeatureCache


class DAGFeatures(FeatureCache):

    def convo_messages(self, conv):
        return self.wrap(conv.convo_id, 'messages', convo_messages, conv=conv)

    def convo_connections(self, conv, check=True):
        return self.wrap(conv.convo_id, 'connections', convo_connections, conv=conv, check=check)

    def convo_user_count(self, conv):
        return self.wrap(conv.convo_id, 'user_cnt', convo_user_count, conv=conv)

    def convo_messages_per_user(self, conv):
        return self.wrap(conv.convo_id, 'post_per_user', convo_messages_per_user, conv=conv)

    def convo_density(self, conv):
        return self.wrap(conv.convo_id, 'nx.density', convo_density, conv=conv)


def convo_messages(conv):
    """
    Returns the number of messages in the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    Returns
    -------
    int
        The number of messages contained in this collection
    """
    return len(conv.posts)


def convo_connections(conv, check=True):
    """
    Returns the number of message reply connections in the Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    check : bool
        Whether to check if a reply_to ID exists in the Conversation. If True, connections without associated
        are not counted.

    Returns
    -------
    int
        The number of message reply connections contained in this collection
    """
    return len([1 for p in conv.posts.values() for ix in p.reply_to if (ix in conv.posts or not check)])


def convo_user_count(conv):
    """
    Given a Conversation, returns the # of unique users who created messages within it

    Parameters
    ----------
    conv : Conversation
        A collection of messages

    Returns
    -------
    int
        The number of unique authors (users) of messages in the Conversation
    """
    return len({p.author for p in conv.posts.values()})


def convo_messages_per_user(conv):
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


def convo_density(conv):
    """
    The density of the conversation as a DAG

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    float
        The density of connection with in the conversation

    Notes
    -----
    See for more information: https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.density.html
    """
    return nx.density(conv.as_graph())
