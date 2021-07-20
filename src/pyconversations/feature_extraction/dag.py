from .base import FeatureCache


class DAGFeatures(FeatureCache):

    def convo_messages(self, conv):
        return self.wrap(conv.convo_id, 'messages', convo_messages, conv=conv)

    def convo_connections(self, conv, check=True):
        return self.wrap(conv.convo_id, 'connections', convo_connections, conv=conv, check=check)


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
