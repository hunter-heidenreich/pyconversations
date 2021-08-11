from collections import defaultdict

import numpy as np

from .post import get_bools as post_bools
from .post import get_floats as post_floats
from .post import get_ints as post_ints
from .post_in_conv import get_bools as pic_bools
from .post_in_conv import get_floats as pic_floats
from .post_in_conv import get_ints as pic_ints


class PostVectorizer:

    def __init__(self, normalization=None):
        # Can be None, 'minmax', 'mean', or 'standard'
        self._norm = normalization

        self._stats = defaultdict(dict)

    def fit(self, posts=None, convs=None, conv=None):
        """
        Fits the parameters necessary for normalization and vectorization of posts.

        Parameters
        ----------
        posts : List(UniMessage)
        convs : List(Conversation)
        conv : Conversation

        Returns
        -------
        None
        """
        if posts:
            self._fit_by_posts(posts)
        elif convs:
            self._fit_by_convs(convs)
        elif conv:
            self._fit_by_convs([conv])
        else:
            raise ValueError

    def _fit_params(self, values):
        """
        Given a dictionary of string keys and numeric values,
        fits the parameters for the desired normalization.

        Parameters
        ----------
        values : dict(str, List(numeric))

        Returns
        -------
        None
        """
        if self._norm is None:
            return
        elif self._norm == 'minmax':
            for k, vs in values.items():
                self._stats[k]['min'] = np.min(vs)
                self._stats[k]['max'] = np.max(vs)
                self._stats[k]['range'] = self._stats[k]['max'] - self._stats[k]['min']
        elif self._norm == 'mean':
            for k, vs in values.items():
                self._stats[k]['min'] = np.min(vs)
                self._stats[k]['max'] = np.max(vs)
                self._stats[k]['range'] = self._stats[k]['max'] - self._stats[k]['min']
                self._stats[k]['mean'] = np.mean(vs)
        elif self._norm == 'standard':
            for k, vs in values.items():
                self._stats[k]['mean'] = np.mean(vs)
                self._stats[k]['std'] = np.std(vs)
        else:
            raise ValueError

    def _fit_by_posts(self, posts):
        """
        Fits the parameters for standardization based on an arbitrary collection of posts
        without their conversations for context

        Parameters
        ----------
        posts : List(UniMessage)

        Returns
        -------
        None
        """
        funcs = [post_floats, post_ints]

        vals = defaultdict(list)
        for post in posts:
            for f in funcs:
                for k, v in f(post).items():
                    vals[k].append(v)

        self._fit_params(vals)

    def _fit_by_convs(self, convs):
        """
        Fits parameters for normalization using conversation information
        for post feature extraction.

        Parameters
        ----------
        convs : List(Conversation)

        Returns
        -------
        None
        """
        funcs = [pic_floats, pic_ints]

        vals = defaultdict(list)
        for conv in convs:
            for post in conv.posts.values():
                for f in funcs:
                    for k, v in f(post, conv).items():
                        vals[k].append(v)

        self._fit_params(vals)

    def transform(self, posts=None, convs=None, conv=None):
        """
        Transforms posts into a a collection of vectors.
        Will perform this extraction with or without conversational features
        depending on provided input.

        Parameters
        ----------
        posts : List(UniMessage)
        convs : List(Conversation)
        conv : Conversation

        Returns
        -------
        np.array
            (N, d), where N is the number of posts and d is the number of features
        """
        if posts:
            return self._transform_by_posts(posts)

        if convs:
            return self._transform_by_convs(convs)

        if conv:
            return self._transform_by_convs([conv])

        raise ValueError

    def _transform_by_posts(self, posts):
        out = []

        funcs = [post_floats, post_ints]
        for post in posts:
            vec = []
            for f in funcs:
                vs = [self._normalize(k, v) for k, v in sorted(f(post).items(), key=lambda kv: kv[0])]
                vec.extend(vs)

            for _, v in sorted(post_bools(post).items(), key=lambda kv: kv[0]):
                vec.append(1 if v else 0)

            out.append(np.array(vec))

        return np.array(out)

    def _transform_by_convs(self, convs):
        out = []

        funcs = [pic_floats, pic_ints]
        for conv in convs:
            for post in conv.posts.values():
                vec = []
                for f in funcs:
                    vs = [self._normalize(k, v) for k, v in sorted(f(post, conv).items(), key=lambda kv: kv[0])]
                    vec.extend(vs)

                for _, v in sorted(pic_bools(post, conv).items(), key=lambda kv: kv[0]):
                    vec.append(1 if v else 0)

                out.append(np.array(vec))

        return np.array(out)

    def _normalize(self, key, value):
        """
        Normalizes a particular key-value pair based on the fit values

        Parameters
        ----------
        key : str
        value : int or float

        Returns
        -------
        float
            The normalized value
        """
        if self._norm is None:
            return value
        elif self._norm == 'minmax':
            if self._stats[key]['range']:
                return (value - self._stats[key]['min']) / self._stats[key]['range']
            else:
                return value
        elif self._norm == 'mean':
            if self._stats[key]['range']:
                return (value - self._stats[key]['mean']) / self._stats[key]['range']
            else:
                return value
        elif self._norm == 'standard':
            if self._stats[key]['std']:
                return (value - self._stats[key]['mean']) / self._stats[key]['std']
            else:
                return value
        else:
            raise ValueError

    def fit_transform(self, posts=None, convs=None, conv=None):
        self.fit(posts, convs, conv)
        return self.transform(posts, convs, conv)
