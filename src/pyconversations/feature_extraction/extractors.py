from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import numpy as np

from ..convo import Conversation
from .conv import get_floats as conv_floats
from .conv import get_ints as conv_ints
from .post import get_bools as post_bools
from .post import get_floats as post_floats
from .post import get_ints as post_ints
from .post_in_conv import get_bools as pic_bools
from .post_in_conv import get_floats as pic_floats
from .post_in_conv import get_ints as pic_ints
from .user_in_conv import collapse_convos
from .user_in_conv import get_bools as user_bools
from .user_in_conv import get_floats as user_floats
from .user_in_conv import get_ints as user_ints
from .user_in_conv import iter_over_users


class Vectorizer(ABC):

    """
    Abstract vectorization class.
    Implements normalization.
    """

    def __init__(self, normalization):
        self._stats = defaultdict(dict)

        # Can be None, 'minmax', 'mean', or 'standard'
        self._norm = normalization

    @abstractmethod
    def fit(self, **kwargs):
        """
        Abstract method for fitting normalization and vectorization parameters
        to data in `kwargs`

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        Vectorizer
            This object should return itself
        """
        pass

    @abstractmethod
    def transform(self, **kwargs):
        """
        Abstract method for transforming data into a vector (or vectors)

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        np.ndarray
        """
        pass

    def fit_transform(self, **kwargs):
        """
        Applies both the fit and transform steps of vectorizer

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        np.ndarray
        """
        return self.fit(**kwargs).transform(**kwargs)

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
                self._stats[k]['min'] = np.nanmin(vs)
                self._stats[k]['max'] = np.nanmax(vs)
                self._stats[k]['range'] = self._stats[k]['max'] - self._stats[k]['min']
        elif self._norm == 'mean':
            for k, vs in values.items():
                self._stats[k]['min'] = np.nanmin(vs)
                self._stats[k]['max'] = np.nanmax(vs)
                self._stats[k]['range'] = self._stats[k]['max'] - self._stats[k]['min']
                self._stats[k]['mean'] = np.nanmean(vs)
        elif self._norm == 'standard':
            for k, vs in values.items():
                self._stats[k]['mean'] = np.nanmean(vs)
                self._stats[k]['std'] = np.nanstd(vs)
        else:
            raise ValueError

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


class PostVectorizer(Vectorizer):

    """
    Vectorization engine for social media post featurization
    """

    def __init__(self, normalization=None, include_conversation=False, include_user=False):
        """
        Constructor for PostVectorizer

        Parameters
        ----------
        normalization : None or str
            Can be None, 'minmax', 'mean', or 'standard'
        include_conversation : bool
        include_user : bool
        """
        super(PostVectorizer, self).__init__(normalization)

        self._include_conversation = include_conversation
        self._include_user = include_user

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
        PostVectorizer
        """
        if posts:
            return self._fit_by_posts(posts)
        elif convs:
            return self._fit_by_convs(convs)
        elif conv:
            return self._fit_by_convs([conv])
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
        PostVectorizer
        """
        funcs = [post_floats, post_ints]

        vals = defaultdict(list)
        for post in posts:
            for f in funcs:
                for k, v in f(post).items():
                    vals[k].append(v)

        if self._include_user:
            cx = Conversation(posts={post.uid: post for post in posts})
            ufs = [user_floats, user_ints]
            for user in iter_over_users(cx):
                for f in ufs:
                    for k, v in f(user, cx).items():
                        vals['user_' + k].append(v)

        self._fit_params(vals)

        return self

    def _fit_by_convs(self, convs):
        """
        Fits parameters for normalization using conversation information
        for post feature extraction.

        Parameters
        ----------
        convs : List(Conversation)

        Returns
        -------
        PostVectorizer
        """
        funcs = [pic_floats, pic_ints]

        vals = defaultdict(list)
        for conv in convs:
            for post in conv.posts.values():
                for f in funcs:
                    for k, v in f(post, conv).items():
                        vals[k].append(v)

            if self._include_conversation:
                conv_fs = [conv_floats, conv_ints]
                for f in conv_fs:
                    for k, v in f(conv).items():
                        vals['convo_' + k].append(v)

        if self._include_user:
            cx = collapse_convos(convs)
            ufs = [user_floats, user_ints]
            for user in iter_over_users(cx):
                for f in ufs:
                    for k, v in f(user, cx).items():
                        vals['user_' + k].append(v)

        self._fit_params(vals)

        return self

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
        """
        Transforms a collection of posts into vectors
        based on fit parameters

        Parameters
        ----------
        posts : List(UniMessage)

        Returns
        -------
        np.array
        """
        out = []
        cx = Conversation(posts={post.uid: post for post in posts})

        funcs = [post_floats, post_ints]
        for post in posts:
            vec = []
            for f in funcs:
                vs = [self._normalize(k, v) for k, v in sorted(f(post).items(), key=lambda kv: kv[0])]
                vec.extend(vs)

            for _, v in sorted(post_bools(post).items(), key=lambda kv: kv[0]):
                vec.append(1 if v else 0)

            if self._include_user:
                ufs = [user_floats, user_ints]
                for f in ufs:
                    vs = [self._normalize('user_' + k, v) for k, v in sorted(f(post.author, cx).items(), key=lambda kv: kv[0])]
                    vec.extend(vs)

                for _, v in sorted(user_bools(post.author, cx).items(), key=lambda kv: kv[0]):
                    vec.append(1 if v else 0)

            out.append(np.array(vec))

        return np.array(out)

    def _transform_by_convs(self, convs):
        """
        Transforms a collection of conversations into vectors for each post
        based on fit parameters

        Parameters
        ----------
        convs : List(Conversation)

        Returns
        -------
        np.array
        """
        out = []
        cx = collapse_convos(convs)

        funcs = [pic_floats, pic_ints]
        for conv in convs:
            for post in conv.posts.values():
                vec = []
                for f in funcs:
                    vs = [self._normalize(k, v) for k, v in sorted(f(post, conv).items(), key=lambda kv: kv[0])]
                    vec.extend(vs)

                for _, v in sorted(pic_bools(post, conv).items(), key=lambda kv: kv[0]):
                    vec.append(1 if v else 0)

                if self._include_conversation:
                    conv_fs = [conv_floats, conv_ints]
                    for f in conv_fs:
                        vs = [self._normalize('convo_' + k, v) for k, v in sorted(f(conv).items(), key=lambda kv: kv[0])]
                        vec.extend(vs)

                if self._include_user:
                    ufs = [user_floats, user_ints]
                    for f in ufs:
                        vs = [self._normalize('user_' + k, v) for k, v in
                              sorted(f(post.author, cx).items(), key=lambda kv: kv[0])]
                        vec.extend(vs)

                    for _, v in sorted(user_bools(post.author, cx).items(), key=lambda kv: kv[0]):
                        vec.append(1 if v else 0)

                out.append(np.array(vec))

        return np.array(out)


class ConversationVectorizer(Vectorizer):

    """
    Vectorization engine for social media conversation featurization
    """

    def __init__(self, normalization=None, agg_post_fts=False, agg_user_fts=False, include_source_user=True):
        """
        Constructor for ConversationVectorizer

        Parameters
        ----------
        normalization : None or str
            Can be None, 'minmax', 'mean', or 'standard'
        agg_post_fts : bool
            Get aggregated measures across posts grouped by disjoint conversations
        agg_user_fts : bool
            Get aggregated measures across users grouped by disjoint conversations
        include_source_user : bool
            Associate user vector of source author with conversation vector
        """
        super(ConversationVectorizer, self).__init__(normalization)

        self._agg_post_fts = agg_post_fts
        self._agg_user_fts = agg_user_fts
        self._include_source_user = include_source_user

    def fit(self, conv=None, convs=None):
        """
        Fits the normalization parameters

        Parameters
        ----------
        conv : Conversation
        convs : List(Conversation)

        Returns
        -------
        ConversationVectorizer
        """
        if conv is not None:
            return self.fit(convs=[conv])
        elif convs is not None:
            vals = defaultdict(list)
            for conv in convs:
                for k, v in conv_floats(conv, include_post=self._agg_post_fts, include_user=self._agg_user_fts).items():
                    vals[k].append(v)

                for k, v in conv_ints(conv).items():
                    vals[k].append(v)

                if self._include_source_user:
                    pid = list(conv.get_sources())[0]
                    user = conv.posts[pid].author

                    fs = [user_floats, user_ints]
                    for f in fs:
                        for k, v in f(user, conv).items():
                            vals['source_user_' + k].append(v)

            self._fit_params(vals)

            return self
        else:
            raise ValueError()

    def transform(self, conv=None, convs=None):
        """
        Returns a set of vectors, one for each supplied conversation.

        Parameters
        ----------
        conv : Conversation
        convs : List(Conversation)

        Returns
        -------
        np.ndarray
        """
        if conv is not None:
            return self.transform(convs=[conv])
        elif convs is not None:
            out = []

            for conv in convs:
                vec = []

                for k, v in sorted(
                    conv_floats(conv, include_post=self._agg_post_fts, include_user=self._agg_user_fts).items(),
                    key=lambda kv: kv[0]
                ):
                    vec.append(self._normalize(k, v))

                for k, v in sorted(conv_ints(conv).items(), key=lambda kv: kv[0]):
                    vec.append(self._normalize(k, v))

                if self._include_source_user:
                    pid = list(conv.get_sources())[0]
                    user = conv.posts[pid].author

                    fs = [user_floats, user_ints]
                    for f in fs:
                        for k, v in sorted(f(user, conv).items(), key=lambda kv: kv[0]):
                            vec.append(self._normalize('source_user_' + k, v))

                out.append(np.array(vec))

            return np.array(out)
        else:
            raise ValueError


class UserVectorizer(Vectorizer):

    """
    Vectorizer for creating user parameter vectors
    """

    def __init__(self, normalization=None, agg_post_fts=False):
        """
        Constructor for UserVectorizer

        Parameters
        ----------
        normalization : None or str
            Can be None, 'minmax', 'mean', or 'standard'
        agg_post_fts : bool
        """
        super(UserVectorizer, self).__init__(normalization)

        self._agg_post_fts = agg_post_fts

    def fit(self, conv=None, convs=None):
        """
        Fits normalization parameters

        Parameters
        ----------
        conv : Conversation
        convs : List(Conversation)

        Returns
        -------
        UserVectorizer
        """
        if conv is not None:
            return self.fit(convs=[conv])
        elif convs is not None:
            vals = defaultdict(list)
            cxs = collapse_convos(convs)
            for user in iter_over_users(cxs):
                for k, v in user_floats(user, cxs, include_post=self._agg_post_fts).items():
                    vals[k].append(v)

                for k, v in user_ints(user, cxs).items():
                    vals[k].append(v)

            self._fit_params(vals)

            return self
        else:
            raise ValueError()

    def transform(self, conv=None, convs=None):
        """
        Returns a set of user vectors for each unique user found

        Parameters
        ----------
        conv : Conversation
        convs : List(Conversation)

        Returns
        -------
        np.ndarray
        """
        if conv is not None:
            return self.transform(convs=[conv])
        elif convs is not None:
            out = []
            cx = collapse_convos(convs)
            for user in iter_over_users(cx):
                vec = []
                for k, v in sorted(
                    user_floats(user, cx, include_post=self._agg_post_fts).items(), key=lambda kv: kv[0]
                ):
                    vec.append(self._normalize(k, v))

                for k, v in sorted(user_ints(user, cx).items(), key=lambda kv: kv[0]):
                    vec.append(self._normalize(k, v))

                out.append(np.array(vec))

            return np.array(out)
        else:
            raise ValueError