from collections import Counter
from functools import reduce

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
        return self.wrap(post.uid, 'types', post_types, post=post)

    def post_type_len(self, post):
        return self.wrap(post.uid, 'type_len', post_type_len, post=post, cache=self)

    def convo_chars(self, conv):
        total, _ = self.wrap(conv.convo_id, 'chars', convo_chars, conv=conv, cache=self)
        return total

    def convo_chars_per_post(self, conv):
        _, dist = self.wrap(conv.convo_id, 'chars', convo_chars, conv=conv, cache=self)
        return dist

    def convo_tokens(self, conv):
        total, _ = self.wrap(conv.convo_id, 'tokens', convo_tokens, conv=conv, cache=self)
        return total

    def convo_tokens_per_post(self, conv):
        _, dist = self.wrap(conv.convo_id, 'tokens', convo_tokens, conv=conv, cache=self)
        return dist

    def convo_token_dist(self, conv):
        return self.wrap(conv.convo_id, 'token_dist', convo_token_dist, conv=conv, cache=self)

    def convo_types(self, conv):
        return self.wrap(conv.convo_id, 'types', convo_types, conv=conv, cache=self)

    def convo_types_per_post(self, conv):
        return self.wrap(conv.convo_id, 'types_per_post', convo_types_per_post, conv=conv, cache=self)


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


def post_type_len(post, cache=None):
    """
    Given a post, returns the length of its text
    in unique token types.

    Parameters
    ----------
    post : UniMessage
        The message to extract features from

    cache : TextFeatures
        An option cache object, which will be used, if specified

    Returns
    -------
    int
        Number of unique tokens in this post
    """
    return len(cache.post_types(post) if cache else post_types(post))


def convo_chars(conv, cache=None):
    """
    Returns the total length of a conversation/collection of posts
    and the size distribution of posts based on the number of characters.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    cache : TextFeatures
        An option cache object, which will be used, if specified

    Returns
    -------
    2-tuple(int, Counter)
        The total aggregate size of the conversation (in chars) as well as the distribution per post
    """
    cnt = Counter([cache.post_char_len(p) if cache else post_char_len(p) for p in conv.posts.values()])

    return sum(map(lambda kv: kv[0] * kv[1], cnt.items())), cnt


def convo_tokens(conv, cache=None):
    """
    Returns the total length of a conversation/collection of posts
    and the size distribution of posts based on the number of tokens.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    cache : TextFeatures
        An option cache object, which will be used, if specified

    Returns
    -------
    2-tuple(int, Counter)
        The total aggregate size of the conversation (in tokens) as well as the distribution per post
    """
    cnt = Counter([cache.post_tok_len(p) if cache else post_tok_len(p) for p in conv.posts.values()])

    return sum(map(lambda kv: kv[0] * kv[1], cnt.items())), cnt


def convo_token_dist(conv, cache=None):
    """
    Returns the token frequency distribution as a Counter for the entire conversation

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    cache : TextFeatures
        An option cache object, which will be used, if specified

    Returns
    -------
    Counter
        The token frequency distribution for the entire conversation
    """
    return reduce(lambda x, y: x + y, [cache.post_tok_dist(p) if cache else post_tok_dist(p) for p in conv.posts.values()])


def convo_types(conv, cache=None):
    """
    Returns the number of unique token types in this conversation.

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    cache : TextFeatures
        An option cache object, which will be used, if specified

    Returns
    -------
    int
        Number of unique token types in this Conversation
    """
    return len(cache.convo_token_dist(conv) if cache else convo_token_dist(conv))


def convo_types_per_post(conv, cache=None):
    """
    Returns the size distribution of posts based on the number of unique tokens (types).

    Parameters
    ----------
    conv : Conversation
        Some collection of posts

    cache : TextFeatures
        An option cache object, which will be used, if specified

    Returns
    -------
    Counter
        The per-post type distribution within this conversation
    """
    return Counter([cache.post_type_len(p) if cache else post_type_len(p) for p in conv.posts.values()])
