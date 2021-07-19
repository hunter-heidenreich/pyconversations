import re

URL_REGEX = re.compile(r'(\b(https?|ftp|file)://)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')


def post_urls(post):
    """
    Given a post, extracts the URLs mentioned within its text

    Parameters
    ----------
    post : UniMessage
        The post for feature extraction

    Returns
    -------
    2-tuple(int, list(str))
        The # of URLs contained in the post and a list of those URLs
    """
    urls = URL_REGEX.findall(post.text)

    return len(urls), urls


def post_user_mentions(post):
    """
    Given a post, extracts the users mentioned within its text

    Parameters
    ----------
    post : UniMessage
        The post for feature extraction

    Returns
    -------
    2-tuple(int, list(str))
        The # of user mentions contained in the post and a list of those mentions
    """
    if post.MENTION_REGEX is None:
        return 0, []

    mentions = re.findall(post.MENTION_REGEX, post.text)

    return len(mentions), mentions
