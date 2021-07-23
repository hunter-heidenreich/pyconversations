import re

from ..utils import memoize

URL_REGEX = re.compile(r'(\b(https?|ftp|file)://)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')


@memoize
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
    urls = [x.group() for x in URL_REGEX.finditer(post.text)]

    return len(urls), urls


def post_url_cnt(post):
    """
    Returns just the URL count

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    int
    """
    return post_urls(post=post)[0]


@memoize
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

    mentions = [x.group() for x in re.finditer(post.MENTION_REGEX, post.text)]

    return len(mentions), mentions


def post_mention_cnt(post):
    """
    Returns just the mention count

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    int
    """
    return post_user_mentions(post=post)[0]


def post_regex_findall_count(post, regex):
    """
    Given a regex pattern, counts all instances of it within the text of a post.

    Parameters
    ----------
    post : UniMessage
    regex : str

    Returns
    -------
    int
    """
    return len(re.findall(regex, post.text))


@memoize
def post_uppercase_count(post):
    """
    Returns a count of the uppercase characters in the text of this post

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    int
    """
    return post_regex_findall_count(post, r'[A-Z]')


@memoize
def post_lowercase_count(post):
    """
    Returns a count of the lowercase characters in the text of this post

    Parameters
    ----------
    post : UniMessage

    Returns
    -------
    int
    """
    return post_regex_findall_count(post, r'[a-z]')
