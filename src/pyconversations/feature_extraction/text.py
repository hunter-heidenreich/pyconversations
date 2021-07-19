from collections import Counter

from .base import FeatureCache


class TextFeatures(FeatureCache):

    """
    Feature extraction engine for descriptive (and frequentist) features to extract from
    the text of messages and conversations.
    """

    def post_char_len(self, post):
        return self.wrap(post.uid, 'char_len', post_char_len, post=post)

    def post_tok_len(self, post):
        return self.wrap(post.uid, 'tok_len', post_tok_len, post=post)

    def post_tok_dist(self, post):
        return self.wrap(post.uid, 'tok_dist', post_tok_dist, post=post)

    def post_types(self, post):
        x = self.get(post.uid, 'types')

        if x is None:
            tok_dist = self.get(post.uid, 'tok_dist')

            if tok_dist:
                x = set(tok_dist.keys())
            else:
                x = post_types(post)

            self.cache(post.uid, 'types', x)

        return x

    def post_type_len(self, post):
        x = self.get(post.uid, 'type_len')

        if x is None:
            x = len(self.post_types(post))
            self.cache(post.uid, 'type_len', x)

        return x


def post_char_len(post):
    """
    Given a post, returns the length of its text
    in characters.

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    int
        The number of characters in this post
    """
    return len(post.text)


def post_tok_len(post):
    """
    Given a post, returns the length of its text
    in tokens (per its tokenizer).

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    int
        The number of tokens in this post
    """
    return len(post.tokens)


def post_tok_dist(post):
    """
    The unigram frequency distribution of tokens within this message

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    Counter
        A counter of the types and the number of times they occur
    """
    return Counter(post.tokens)


def post_types(post):
    """
    Returns the set of unique tokens contained within the text of the post

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    set(str)
        The set of unique token types
    """
    return set(post.tokens)


def post_type_len(post):
    """
    Given a post, returns the length of its text
    in unique token types.

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    Returns
    -------
    int
        Number of unique tokens in this post
    """
    return len(post_types(post))
