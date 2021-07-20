from .base import UniMessage
from .chan import ChanPost
from .facebook import FBPost
from .reddit import RedditPost
from .twitter import Tweet


def get_constructor_by_platform(key):
    """
    Returns the UniMessage constructor associated with a platform

    Parameters
    ----------
    key : str
        A platform name

    Returns
    -------
    UniMesssage Constructor
        The constructor for the UniMessage associated with `key`
    """
    return {
        '4chan':    ChanPost,
        'Facebook': FBPost,
        'Reddit':   RedditPost,
        'Twitter':  Tweet,
    }[key]


__all__ = [
    'UniMessage',
    'Tweet',
    'FBPost',
    'RedditPost',
    'ChanPost',
    'get_constructor_by_platform'
]
