def sum_bools_in_conversation(conv, check_fn):
    """
    Iterates across the posts in `conv` and counts the number
    of posts with the property checked in `check_fn`

    Parameters
    ----------
    conv : Conversation
        The conversation to iterate over

    check_fn : lambda : (UniMessage, Conversation) -> bool
        A lambda function that evaluates a property

    Returns
    -------
    int
        The number of posts in the conversation with the property
    """
    n = 0

    for p in conv.posts.values():
        if check_fn(post=p, conv=conv):
            n += 1

    return n


def memoize(func):

    memo = dict()

    def wrap(**kwargs):
        if 'post' in kwargs and 'conv' in kwargs:
            uid = merge_ids(kwargs['post'], kwargs['conv'])
        elif 'post' in kwargs:
            uid = kwargs['post'].uid
        elif 'conv' in kwargs:
            uid = kwargs['conv'].convo_id
        else:
            uid = -1

        if uid in memo and not kwargs.get('refresh', False):
            return memo[uid]

        if 'refresh' in kwargs:
            del kwargs['refresh']

        val = func(**kwargs)
        memo[uid] = val

        return val

    return wrap


def merge_ids(post, convo):
    """
    Merges the identifiers of a post and conversation.

    Parameters
    ----------
    post : UniMessage
        A message
    convo : Conversation
        A collection of messages

    Returns
    -------
    str
        A new, merged ID
    """
    return str(post.uid) + '_' + str(convo.convo_id)
