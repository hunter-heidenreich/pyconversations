from collections import Counter

from ..utils import memoize


@memoize
def convo_in_degrees(conv):
    """
    Returns a Counter of the post IDs mapping to the # of replies that post received in this Conversation

    Parameters
    ----------
    conv : Conversation
        A collection of posts

    Returns
    -------
    Counter
        A mapping from post IDs to the # of replies they receive in `conv`
    """
    cnt = Counter()
    for p in conv.posts.values():
        if p.uid not in cnt:
            cnt[p.uid] = 0

        for r in p.reply_to:
            if r in conv.posts:
                cnt[r] += 1
    return cnt
