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
from .dag import convo_sources
from .dag import convo_tree_degree
from .dag import convo_tree_depth
from .dag import convo_tree_width
from .dag import convo_user_count
from .dag import convo_user_size_dist
from .regex import convo_mention_stats
from .regex import convo_url_stats
from .regex import post_urls
from .regex import post_user_mentions
from .temporal import convo_duration
from .temporal import convo_post_ages
from .temporal import convo_reply_times
from .text import convo_char_stats
from .text import convo_token_dist
from .text import convo_token_stats
from .text import convo_type_stats
from .text import post_char_len
from .text import post_tok_dist
from .text import post_tok_len
from .text import post_type_len


class PostFeatures:

    """
    For extracting features from posts in isolation
    """

    @staticmethod
    def get_all(post):
        """
        Returns all features as a nested dictionary.

        Parameters
        ----------
        post : UniMessage
            The message to extract features from

        Returns
        -------
        dict()
            A map from feature name to value
        """
        return {
            'id':     post.uid,
            'author': post.author,
            'counts': PostFeatures.get_counts(post),
            'dists':  PostFeatures.get_distributions(post)
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
        dict(str, int)
            A map from feature name to value
        """
        return {
            'urls':     post_urls(post=post)[0],
            'mentions': post_user_mentions(post=post)[0],
            'chars':    post_char_len(post=post),
            'tokens':   post_tok_len(post=post),
            'types':    post_type_len(post=post)
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
        dict(str, dict(Hashable, int))
            A map from feature name to value
        """
        return {
            'unigrams': post_tok_dist(post=post)
        }


class ConvFeatures:

    @staticmethod
    def get_all(conv):
        return {
            'id':     conv.convo_id,
            'counts': ConvFeatures.get_counts(conv),
            'dists':  ConvFeatures.get_distributions(conv),
            'ratios': ConvFeatures.get_ratios(conv),
            'nums':   ConvFeatures.get_nums(conv),
        }

    @staticmethod
    def get_counts(conv):
        return {
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

    @staticmethod
    def get_nums(conv):
        """
        Unbounded numerical features
        :param conv:
        :return:
        """
        out = {
            'duration': convo_duration(conv=conv),
        }

        for key, fn in [
            ('mentions', convo_mention_stats),
            ('urls', convo_url_stats),
            ('reply_times', convo_reply_times),
            ('post_ages', convo_post_ages),
            ('chars', convo_char_stats),
            ('tokens', convo_token_stats),
            ('types', convo_type_stats),
        ]:
            for k, v in fn(conv=conv).items():
                if k == 'total':
                    continue

                out[key + '_' + k] = v

        return out

    @staticmethod
    def get_distributions(conv):
        return {
            'in_degree':         convo_in_degrees_dist(conv=conv),
            'out_degree':        convo_out_degrees_dist(conv=conv),
            'degree':            convo_degrees_dist(conv=conv),
            'depths':            convo_depth_dist(conv=conv),
            'messages_per_user': convo_messages_per_user(conv=conv),
            'unigrams':          convo_token_dist(conv=conv),
            'user_size':         convo_user_size_dist(conv=conv),
        }

    @staticmethod
    def get_ratios(conv):
        """
        Extracts conversation-level ratio features.
        Features are guaranteed to be in the 0-1 range.

        Parameters
        ----------
        conv : Conversation

        Returns
        -------
        dict
        """
        return {
            'density':  convo_density(conv=conv),
            'source':   convo_sources(conv=conv) / convo_messages(conv=conv),
            'leaf':     convo_leaves(conv=conv) / convo_messages(conv=conv),
            'internal': convo_internal_nodes(conv=conv) / convo_messages(conv=conv),
        }
