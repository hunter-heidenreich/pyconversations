from collections import defaultdict

import numpy as np

from .dag import convo_connections
from .dag import convo_density
from .dag import convo_internal_nodes
from .dag import convo_leaves
from .dag import convo_messages
from .dag import convo_post_depth
from .dag import convo_post_width
from .dag import convo_sources
from .dag import convo_tree_degree
from .dag import convo_tree_depth
from .dag import convo_tree_width
from .dag import convo_user_count
from .dag import convo_users_posts_in_convo
from .dag import is_post_internal_node
from .dag import is_post_leaf
from .dag import is_post_source
from .dag import post_degree
from .dag import post_in_degree
from .dag import post_out_degree
from .regex import convo_mention_stats
from .regex import convo_url_stats
from .regex import post_lowercase_count
from .regex import post_uppercase_count
from .regex import post_urls
from .regex import post_user_mentions
from .temporal import convo_duration
from .temporal import post_reply_time
from .temporal import post_to_source
from .text import avg_token_entropy_all_splits
from .text import convo_char_stats
from .text import convo_token_stats
from .text import convo_type_stats
from .text import post_char_len
from .text import post_tok_len
from .text import post_type_len


class PostFeaturizer:

    def __init__(self, binary=True, numeric=True, categorical=True):
        self._binary = binary
        self._numeric = numeric
        self._categorical = categorical

    def transform(self, post, conv=None):
        features = {
            'id': post.uid,
        }

        if conv is not None:
            features['convo_id'] = conv.convo_id

        if self._binary:
            features['binary'] = self.transform_binary(post, conv)

        if self._numeric:
            features['numeric'] = self.transform_numeric(post, conv)

        if self._categorical:
            features['categorical'] = self.transform_categorical(post, conv)

        return features

    def transform_binary(self, post, conv=None):
        ft = {}

        if conv is not None:
            ft_in_conv = {
                'is_source':   1 if is_post_source(post=post, conv=conv) else 0,
                'is_leaf':     1 if is_post_leaf(post=post, conv=conv) else 0,
                'is_internal': 1 if is_post_internal_node(post=post, conv=conv) else 0,
            }
            ft = {**ft, **ft_in_conv}

        return ft

    def transform_numeric(self, post, conv=None):
        ft = {
            'urls':      post_urls(post=post)[0],
            'mentions':  post_user_mentions(post=post)[0],
            'chars':     post_char_len(post=post),
            'tokens':    post_tok_len(post=post),
            'types':     post_type_len(post=post),
            'uppercase': post_uppercase_count(post=post),
            'lowercase': post_lowercase_count(post=post),
        }
        ft['uppercase_ratio'] = ft['uppercase'] / (ft['uppercase'] + ft['lowercase']) if ft['uppercase'] or ft[
            'lowercase'] else 0

        if conv is not None:
            ft_in_conv = {
                'depth':        convo_post_depth(post=post, conv=conv),
                'width':        convo_post_width(post=post, conv=conv),
                'degree':       post_degree(post=post, conv=conv),
                'in_degree':    post_in_degree(post=post, conv=conv),
                'out_degree':   post_out_degree(post=post, conv=conv),
                'author_posts': convo_users_posts_in_convo(post=post, conv=conv),
                'age':          post_to_source(post=post, conv=conv),
                'reply_time':   post_reply_time(post=post, conv=conv),
            }

            for k, v in avg_token_entropy_all_splits(post=post, conv=conv).items():
                ft_in_conv['avg_token_entropy_' + k] = v

            ft = {**ft, **ft_in_conv}

        return ft

    def transform_categorical(self, post, conv=None):
        ft = {
            'platform': post.platform,
            'lang':     post.lang,
            'author':   post.author,
        }

        return ft


class ConversationFeaturizer:

    def __init__(self, binary=True, numeric=True, categorical=True, include_post=True):
        self._binary = binary
        self._numeric = numeric
        self._categorical = categorical
        self._include_post = include_post
        self._post_ft = PostFeaturizer(
            binary=binary, numeric=numeric, categorical=categorical
        )

    def transform(self, conv):
        features = {
            'convo_id': conv.convo_id,
        }

        if self._binary:
            features['binary'] = self.transform_binary(conv)

        if self._numeric:
            features['numeric'] = self.transform_numeric(conv)

        if self._categorical:
            features['categorical'] = self.transform_categorical(conv)

        return features

    def transform_binary(self, conv):
        ft = {}

        if self._include_post:
            ft['posts'] = {
                pid: self._post_ft.transform_binary(post, conv)
                for pid, post in conv.posts.items()
            }

        return ft

    def transform_numeric(self, conv):
        ft = {
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

            'duration':        convo_duration(conv=conv),
            'density':         convo_density(conv=conv),

            'source_ratio':    convo_sources(conv=conv) / convo_messages(conv=conv),
            'leaf_ratio':      convo_leaves(conv=conv) / convo_messages(conv=conv),
            'internal_ratio':  convo_internal_nodes(conv=conv) / convo_messages(conv=conv),
            'user_post_ratio': convo_user_count(conv=conv) / convo_messages(conv=conv),
        }

        # I think we can refactor this and transform it into an auto generate
        # via post extraction
        # for key, fn in [
        #     ('mentions', convo_mention_stats),
        #     ('urls', convo_url_stats),
        #     ('reply_times', convo_reply_times),
        #     ('post_ages', convo_post_ages),
        #     ('chars', convo_char_stats),
        #     ('tokens', convo_token_stats),
        #     ('types', convo_type_stats),
        #     ('user_sizes', convo_user_size_stats),
        # ]:
        #     for k, v in fn(conv=conv).items():
        #         if k == 'total':
        #             continue
        #
        #         ft[key + '_' + k] = v

        if self._include_post:
            ft['posts'] = {}

        aggregator = defaultdict(list)
        for pid, post in conv.posts.items():
            fx = self._post_ft.transform_numeric(post, conv)

            for k, v in fx.items():
                aggregator[k].append(v)

            if self._include_post:
                ft['posts'][pid] = fx

        for key, vals in aggregator.items():
            ft[key + '_min'] = np.min(vals)
            ft[key + '_max'] = np.max(vals)
            ft[key + '_mean'] = np.mean(vals)
            ft[key + '_std'] = np.std(vals)
            ft[key + '_median'] = np.median(vals)

        return ft

    def transform_categorical(self, conv):
        ft = {}

        if self._include_post:
            ft['posts'] = {
                pid: self._post_ft.transform_categorical(post, conv)
                for pid, post in conv.posts.items()
            }

        return ft
