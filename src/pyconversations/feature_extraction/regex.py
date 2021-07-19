import re

from .base import FeatureCache

URL_REGEX = re.compile(r'(\b(https?|ftp|file)://)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')


class RegexFeatures(FeatureCache):

    """
    Feature extraction based on regex pattern matching over text.
    """

    def post_url_cnt(self, post):
        cnt, _ = self.wrap(post.uid, 'urls', post_urls, post=post)

        return cnt

    def post_urls(self, post):
        _, urls = self.wrap(post.uid, 'urls', post_urls, post=post)

        return urls

    def post_mention_cnt(self, post):
        cnt, _ = self.wrap(post.uid, 'mentions', post_user_mentions, post=post)

        return cnt

    def post_mentions(self, post):
        _, mentions = self.wrap(post.uid, 'mentions', post_user_mentions, post=post)

        return mentions


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
