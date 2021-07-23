import numpy as np


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


def agg_nums_in_conversation(conv, get_num_fn, use_conv=True):
    """
    Given a conversation and a function that computes a numerical statistic on a post,
    this function returns a dictionary of the total, average, std. dev., median, min, and max.

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    get_num_fn : lambda (post -> number)
        A function that maps a post to a numerical statistic

    use_conv : bool
        Whether the post function needs the conversation as input. Default: True

    Returns
    -------
    dict
        The total, avg., std. dev., median, min, and max of the numerical statistic
    """
    n = None
    dist = []
    for p in conv.posts.values():
        x = get_num_fn(post=p, conv=conv) if use_conv else get_num_fn(post=p)
        dist.append(x)

        if n is None:
            n = x
        else:
            n += x

    return {
        'total':  n,
        'avg':    np.average(dist),
        'std':    np.std(dist),
        'median': np.median(dist),
        'min':    np.min(dist),
        'max':    np.max(dist),
    }


def memoize(func):

    memo = dict()

    def wrap(**kwargs):
        if 'post' in kwargs and 'conv' in kwargs:
            uid = merge_ids(kwargs['post'], kwargs['conv'])
        elif 'post' in kwargs:
            uid = kwargs['post'].uid
        elif 'conv' in kwargs:
            uid = kwargs['conv'].convo_id
        elif 'conv_a' in kwargs and 'conv_b' in kwargs:
            uid = str(kwargs['conv_a']) + '_' + str(kwargs['conv_b'])
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


def merge_dicts(this, that):
    """
    Assuming two dictionaries of the form dict(str, dict(str, Any))
    where the nested dictionaries _do not_ overlap in key space,
    merges these dictionaries.

    Paramters
    ---------
    this : dict(str, dict(str, Any))
    that : dict(str, dict(str, Any))

    Returns
    -------
    dict(str, dict(str, Any))
        The merged dictionary
    """
    for k, v in that.items():
        if k in this and k not in ['id', 'convo_id']:
            this[k] = {**this[k], **that[k]}  # merges two dictionaries, assuming no key overlap)
        else:
            this[k] = v

    return this
