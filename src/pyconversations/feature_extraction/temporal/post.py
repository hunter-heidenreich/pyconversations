from ..utils import memoize


@memoize
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


@memoize
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

    return (post.created_at - conv.posts[timeorder[0]].created_at).total_seconds()
