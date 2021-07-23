from .dag import convo_connections
from .dag import convo_degrees_dist
from .dag import convo_density
from .dag import convo_depth_dist
from .dag import convo_in_degrees_dist
from .dag import convo_internal_nodes
from .dag import convo_leaves
from .dag import convo_messages
from .dag import convo_messages_per_user
from .dag import convo_out_degrees_dist
from .dag import convo_post_depth
from .dag import convo_post_width
from .dag import convo_sources
from .dag import convo_tree_degree
from .dag import convo_tree_depth
from .dag import convo_tree_width
from .dag import convo_user_count
from .dag import convo_user_size_dist
from .dag import convo_user_size_stats
from .dag import convo_users_posts_in_convo
from .dag import is_post_internal_node
from .dag import is_post_leaf
from .dag import is_post_source
from .dag import post_degree
from .dag import post_in_degree
from .dag import post_out_degree
from .regex import convo_mention_stats
from .regex import convo_url_stats
from .regex import post_urls
from .regex import post_user_mentions
from .temporal import convo_duration
from .temporal import convo_post_ages
from .temporal import convo_reply_times
from .temporal import post_reply_time
from .temporal import post_to_source
from .text import avg_token_entropy_all_splits
from .text import convo_char_stats
from .text import convo_token_dist
from .text import convo_token_stats
from .text import convo_type_stats
from .text import post_char_len
from .text import post_tok_dist
from .text import post_tok_len
from .text import post_type_len
from .utils import merge_dicts


class Features:

    @staticmethod
    def get_all(*args, **kwargs):
        return {}

    @staticmethod
    def get_binary(*args, **kwargs):
        return {}

    @staticmethod
    def get_counts(*args, **kwargs):
        return {}

    @staticmethod
    def get_distributions(*args, **kwargs):
        return {}

    @staticmethod
    def get_nums(*args, **kwargs):
        return {}


class PostFeatures(Features):
    """
    For extracting features from posts in isolation
    """

    @staticmethod
    def get_all(post):
        """
        Returns features in a nested dictionary

        Parameters
        ----------
        post : UniMessage
            The message to extract features from

        Returns
        -------
        dict
        """
        return {
            'id':     post.uid,
            'author': post.author,
            **PostFeatures.get_counts(post),
            **PostFeatures.get_distributions(post),
        }

    @staticmethod
    def get_counts(post):
        """
        Given a post, returns all features that are natural numbers (i.e., counts).
        Features are guaranteed to be within the range [0, inf).

        Parameters
        ----------
        post : UniMessage
            The message to extract features from

        Returns
        -------
        dict(str, dict(str, int))
        """
        return {
            'counts': {
                'urls':     post_urls(post=post)[0],
                'mentions': post_user_mentions(post=post)[0],
                'chars':    post_char_len(post=post),
                'tokens':   post_tok_len(post=post),
                'types':    post_type_len(post=post)
            }
        }

    @staticmethod
    def get_distributions(post):
        """
        Given a post, returns all features that are frequency distributions.
        Features are guaranteed to be a dictionary which maps something (e.g., a token type)
        to the amount of times it occurs within this post.

        Parameters
        ----------
        post : UniMessage
            The message to extract features from

        Returns
        -------
        dict(str, dict(str, dict(Hashable, int)))
            A map from feature name to value
        """
        return {
            'dists': {
                'unigrams': post_tok_dist(post=post)
            }
        }


class ConvFeatures(Features):
    """
    For extracting features from full conversations, or collections of posts
    """

    @staticmethod
    def get_all(conv, include_post_features=False, include_static=False, include_token_entropy=True):
        """
        Returns features in a nested dictionary

        Parameters
        ----------
        conv : Conversation
            A collection of posts

        include_post_features : bool
            Whether or not to include post features in output. Default: False

        include_static : bool
            If including posts, whether or not to include their static features. Default: False

        Returns
        -------
        dict
        """
        out = {
            'id': conv.convo_id,
            **ConvFeatures.get_counts(conv),
            **ConvFeatures.get_distributions(conv),
            **ConvFeatures.get_nums(conv),
        }

        if include_post_features:
            out['posts'] = {
                pid: PostInConvFeatures.get_all(
                    conv, post, include_static=include_static, include_token_entropy=include_token_entropy
                )
                for pid, post in conv.posts.items()
            }

        return out

    @staticmethod
    def get_counts(conv, include_post_features=False, include_static=False):
        """
        Returns all features that are natural numbers (i.e., counts).
        Features are guaranteed to be within the range [0, inf).

        Parameters
        ----------
        conv : Conversation
            A collection of posts

        include_post_features : bool
            Whether or not to include post features in output. Default: False

        include_static : bool
            If including posts, whether or not to include their static features. Default: False

        Returns
        -------
        dict
        """
        out = {
            'counts': {
                'chars':          convo_char_stats(conv=conv)['total'],
                'connections':    convo_connections(conv=conv),
                'internal_nodes': convo_internal_nodes(conv=conv),
                'leaves':         convo_leaves(conv=conv),
                'mentions':       convo_mention_stats(conv=conv)['total'],
                'messages':       convo_messages(conv=conv),
                'sources':        convo_sources(conv=conv),
                'tokens':         convo_token_stats(conv=conv)['total'],
                'tree_degree':    convo_tree_degree(conv=conv),
                'tree_depth':     convo_tree_depth(conv=conv),
                'tree_width':     convo_tree_width(conv=conv),
                'types':          convo_type_stats(conv=conv)['total'],
                'users':          convo_user_count(conv=conv),
                'urls':           convo_url_stats(conv=conv)['total'],
            }
        }

        if include_post_features:
            out['posts'] = {
                pid: PostInConvFeatures.get_counts(conv, post, include_static=include_static)
                for pid, post in conv.posts.items()
            }

        return out

    @staticmethod
    def get_nums(conv, include_post_features=False, include_static=False, include_token_entropy=True):
        """
        Unbounded numerical features.

        Parameters
        ----------
        conv : Conversation
            A collection of posts

        include_post_features : bool
            Whether or not to include post features in output. Default: False

        include_static : bool
            If including posts, whether or not to include their static features. Default: False

        Returns
        -------
        dict
        """
        out = {
            'nums': {
                'duration':        convo_duration(conv=conv),
                'density':         convo_density(conv=conv),
                'source_ratio':    convo_sources(conv=conv) / convo_messages(conv=conv),
                'leaf_ratio':      convo_leaves(conv=conv) / convo_messages(conv=conv),
                'internal_ratio':  convo_internal_nodes(conv=conv) / convo_messages(conv=conv),
                'user_post_ratio': convo_user_count(conv=conv) / convo_messages(conv=conv),
            }
        }

        for key, fn in [
            ('mentions', convo_mention_stats),
            ('urls', convo_url_stats),
            ('reply_times', convo_reply_times),
            ('post_ages', convo_post_ages),
            ('chars', convo_char_stats),
            ('tokens', convo_token_stats),
            ('types', convo_type_stats),
            ('user_sizes', convo_user_size_stats),
        ]:
            for k, v in fn(conv=conv).items():
                if k == 'total':
                    continue

                out['nums'][key + '_' + k] = v

        if include_post_features:
            out['posts'] = {
                pid: PostInConvFeatures.get_nums(
                    conv, post, include_static=include_static, include_token_entropy=include_token_entropy
                )
                for pid, post in conv.posts.items()
            }

        return out

    @staticmethod
    def get_distributions(conv, include_post_features=False, include_static=False):
        """
        Returns all features that are frequency distributions.
        Features are guaranteed to be a dictionary which maps something (e.g., a token type)
        to the amount of times it occurs within this post.

        Parameters
        ----------
        conv : Conversation
            A collection of posts

        include_post_features : bool
            Whether or not to include post features in output. Default: False

        include_static : bool
            If including posts, whether or not to include their static features. Default: False

        Returns
        -------
        dict
        """
        out = {
            'dists': {
                'in_degree':         convo_in_degrees_dist(conv=conv),
                'out_degree':        convo_out_degrees_dist(conv=conv),
                'degree':            convo_degrees_dist(conv=conv),
                'depths':            convo_depth_dist(conv=conv),
                'messages_per_user': convo_messages_per_user(conv=conv),
                'unigrams':          convo_token_dist(conv=conv),
                'user_size':         convo_user_size_dist(conv=conv),
            }
        }

        if include_post_features:
            out['posts'] = {
                pid: PostInConvFeatures.get_distributions(conv, post, include_static=include_static)
                for pid, post in conv.posts.items()
            }

        return out


class PostInConvFeatures(Features):

    @staticmethod
    def get_all(conv, post, include_static=False, include_token_entropy=True):
        if include_static:
            return merge_dicts(
                PostInConvFeatures.get_all(conv, post, include_token_entropy=include_token_entropy),
                PostFeatures.get_all(post)
            )

        return {
            'id':       post.uid,
            'convo_id': conv.convo_id,
            **PostInConvFeatures.get_counts(conv, post),
            **PostInConvFeatures.get_binary(conv, post),
            **PostInConvFeatures.get_nums(conv, post, include_token_entropy=include_token_entropy),
        }

    @staticmethod
    def get_counts(conv, post, include_static=False):
        if include_static:
            return merge_dicts(
                PostInConvFeatures.get_counts(conv, post), PostFeatures.get_counts(post)
            )

        return {
            'counts': {
                'depth':        convo_post_depth(post=post, conv=conv),
                'width':        convo_post_width(post=post, conv=conv),
                'degree':       post_degree(post=post, conv=conv),
                'in_degree':    post_in_degree(post=post, conv=conv),
                'out_degree':   post_out_degree(post=post, conv=conv),
                'author_posts': convo_users_posts_in_convo(post=post, conv=conv),
            }
        }

    @staticmethod
    def get_binary(conv, post, include_static=False):
        return {
            'binary': {
                'is_source':   1 if is_post_source(post=post, conv=conv) else 0,
                'is_leaf':     1 if is_post_leaf(post=post, conv=conv) else 0,
                'is_internal': 1 if is_post_internal_node(post=post, conv=conv) else 0,
            }
        }

    @staticmethod
    def get_nums(conv, post, include_static=False, include_token_entropy=True):
        out = {
            'nums': {
                'age':        post_to_source(post=post, conv=conv),
                'reply_time': post_reply_time(post=post, conv=conv),
            }
        }

        if include_token_entropy:
            for k, v in avg_token_entropy_all_splits(post=post, conv=conv).items():
                out['nums']['avg_token_entropy_' + k] = v

        if include_static:
            out = merge_dicts(out, PostFeatures.get_nums(post))

        return out
