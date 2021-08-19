from abc import ABC
from abc import abstractmethod

import numpy as np
from tqdm import tqdm

from ..convo import Conversation
from ..message import UniMessage
from .conv import get_floats as conv_floats
from .conv import get_ints as conv_ints
from .conv import messages_per_user
from .post import get_bools as post_bools
from .post import get_floats as post_floats
from .post import get_ints as post_ints
from .post_in_conv import get_bools as pic_bools
from .post_in_conv import get_floats as pic_floats
from .post_in_conv import get_ints as pic_ints
from .user_across_conv import get_floats as uac_floats
from .user_across_conv import get_ints as uac_ints
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
        self._stats = {}

        # feature name to column index
        self._ft2col = {}
        self._num_bool = 0

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

    def fit_transform(self, *args):
        """
        Applies both the fit and transform steps of vectorizer

        Parameters
        ----------
        args : List

        Returns
        -------
        np.ndarray
        """
        return self.fit(*args).transform(*args)

    def _fit_params(self, values):
        """
        Given a matrix of values,
        fits the parameters for the desired normalization.

        Parameters
        ----------
        values : np.ndarray (2D)

        Returns
        -------
        None
        """
        if self._norm is None:
            return
        elif self._norm == 'minmax':
            self._stats['min'] = np.nanmin(values, axis=0)
            self._stats['range'] = np.nanmax(values, axis=0) - self._stats['min']

            # fix divide issues
            self._stats['range'][self._stats['range'] == 0] = 1
        elif self._norm == 'mean':
            self._stats['range'] = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
            self._stats['mean'] = np.nanmean(values, axis=0)

            # fix divide issues
            self._stats['range'][self._stats['range'] == 0] = 1
        elif self._norm == 'standard':
            self._stats['mean'] = np.nanmean(values, axis=0)
            self._stats['std'] = np.nanstd(values, axis=0)

            # fix divide issues
            self._stats['std'][self._stats['std'] == 0] = 1
        else:
            raise ValueError

    def _normalize(self, values):
        """
        Normalizes the data

        Parameters
        ----------
        values : np.ndarry

        Returns
        -------
        np.ndarry
        """
        if self._norm is None:
            return values
        elif self._norm == 'minmax':
            return (values - self._stats['min']) / self._stats['range']
        elif self._norm == 'mean':
            return (values - self._stats['mean']) / self._stats['range']
        elif self._norm == 'standard':
            return (values - self._stats['mean']) / self._stats['std']
        else:
            raise ValueError


class PostVectorizer(Vectorizer):

    """
    Vectorization engine for social media post featurization
    """

    def __init__(self, normalization=None):
        """
        Constructor for PostVectorizer

        Parameters
        ----------
        normalization : None or str
            Can be None, 'minmax', 'mean', or 'standard'
        """
        super(PostVectorizer, self).__init__(normalization)

    def fit(self, xs):
        """
        Fits the parameters necessary for normalization and vectorization of posts.

        Parameters
        ----------
        xs : List(UniMessage), List(Conversation), Conversation

        Returns
        -------
        PostVectorizer
        """

        if type(xs) == list:
            if isinstance(xs[0], Conversation):
                return self._fit_by_convs(xs)
            elif isinstance(xs[0], UniMessage):
                return self._fit_by_posts(xs)

        elif isinstance(xs, Conversation):
            return self._fit_by_convs([xs])

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
        values = None
        for ix, post in tqdm(enumerate(posts), desc='PostVec: Fitting by posts'):
            params = {**post_floats(post), **post_ints(post)}

            if values is None:
                values = np.zeros((len(posts), len(params)))
                self._ft2col = {}
                self._num_bool = len(post_bools(post))
                for k in params:
                    self._ft2col[k] = len(self._ft2col)

            for k, v in params.items():
                values[ix, self._ft2col[k]] = v

        self._fit_params(values)

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
        values = None
        ix = 0
        total_posts = sum(map(lambda c: len(c.posts), convs))

        for conv in tqdm(convs, desc='PostVec: Fitting by conversations'):
            for post in conv.posts.values():
                params = {**pic_floats(post, conv), **pic_ints(post, conv)}

                if values is None:
                    values = np.zeros((total_posts, len(params)))
                    self._ft2col = {}
                    self._num_bool = len(pic_bools(post, conv))
                    for k in params:
                        self._ft2col[k] = len(self._ft2col)

                for k, v in params.items():
                    values[ix, self._ft2col[k]] = v

                ix += 1

        self._fit_params(values)

        return self

    def transform(self, xs, include_ids=False):
        """
        Transforms posts into a a collection of vectors.
        Will perform this extraction with or without conversational features
        depending on provided input.

        Parameters
        ----------
        xs : List(UniMessage), List(Conversation), Conversation
        include_ids : bool

        Returns
        -------
        np.array
            (N, d), where N is the number of posts and d is the number of features
        """
        if type(xs) == list:
            if isinstance(xs[0], Conversation):
                return self._transform_by_convs(xs, include_ids)
            elif isinstance(xs[0], UniMessage):
                return self._transform_by_posts(xs, include_ids)
        elif isinstance(xs, Conversation):
            return self._transform_by_convs([xs], include_ids)

        raise ValueError

    def _transform_by_posts(self, posts, include_ids):
        """
        Transforms a collection of posts into vectors
        based on fit parameters

        Parameters
        ----------
        posts : List(UniMessage)
        include_ids : bool

        Returns
        -------
        np.array
        """
        ids = {}
        out = np.zeros((len(posts), len(self._ft2col)))
        out_bools = np.zeros((len(posts), self._num_bool))

        funcs = [post_floats, post_ints]
        for ix, post in tqdm(enumerate(posts), desc='PostVec: Transforming by posts'):
            for f in funcs:
                for k, v in f(post).items():
                    out[ix, self._ft2col[k]] = v

            for ik, (_, v) in enumerate(sorted(post_bools(post).items(), key=lambda kv: kv[0])):
                out_bools[ix, ik] = 1 if v else 0

            ids[post.uid] = ix

        out = self._normalize(out)
        out = np.hstack((out, out_bools))

        if include_ids:
            return out, ids

        return out

    def _transform_by_convs(self, convs, include_ids):
        """
        Transforms a collection of conversations into vectors for each post
        based on fit parameters

        Parameters
        ----------
        convs : List(Conversation)
        include_ids : bool

        Returns
        -------
        dict(str, np.array)
        """
        ix = 0
        ids = {}
        total_posts = sum(map(lambda c: len(c.posts), convs))
        out = np.zeros((total_posts, len(self._ft2col)))
        out_bools = np.zeros((total_posts, self._num_bool))

        funcs = [pic_floats, pic_ints]
        for conv in tqdm(convs, desc='PostVec: Transforming by conversations'):
            for post in conv.posts.values():
                for f in funcs:
                    for k, v in f(post, conv).items():
                        out[ix, self._ft2col[k]] = v

                for ik, (_, v) in enumerate(sorted(pic_bools(post, conv).items(), key=lambda kv: kv[0])):
                    out_bools[ix, ik] = 1 if v else 0

                ix += 1
                ids[(conv.convo_id, post.uid)] = ix

        out = self._normalize(out)
        out = np.hstack((out, out_bools))

        if include_ids:
            return out, ids

        return out


class ConversationVectorizer(Vectorizer):

    """
    Vectorization engine for social media conversation featurization
    """

    def __init__(self, normalization=None):
        """
        Constructor for ConversationVectorizer

        Parameters
        ----------
        normalization : None or str
            Can be None, 'minmax', 'mean', or 'standard'
        """
        super(ConversationVectorizer, self).__init__(normalization)

    def fit(self, xs):
        """
        Fits the normalization parameters

        Parameters
        ----------
        xs : Conversation or List(Conversation)

        Returns
        -------
        ConversationVectorizer
        """
        if isinstance(xs, Conversation):
            return self.fit([xs])
        elif type(xs) == list and isinstance(xs[0], Conversation):
            values = None
            for ix, conv in tqdm(enumerate(xs), desc='ConvVec: Fitting by conversations', total=len(xs)):
                params = {**conv_floats(conv), **conv_ints(conv)}

                if values is None:
                    values = np.zeros((len(xs), len(params)))
                    self._ft2col = {}
                    # self._num_bool = len(post_bools(post))
                    for k in params:
                        self._ft2col[k] = len(self._ft2col)

                for k, v in params.items():
                    values[ix, self._ft2col[k]] = v

            self._fit_params(values)

            return self

        raise ValueError()

    def transform(self, xs, include_ids=False):
        """
        Returns a set of vectors, one for each supplied conversation.

        Parameters
        ----------
        xs : Conversation or List(Conversation)
        include_ids : bool

        Returns
        -------
        dict(str, np.array)
        """
        if isinstance(xs, Conversation):
            return self.transform([xs], include_ids=include_ids)
        elif type(xs) == list and isinstance(xs[0], Conversation):
            ids = {}
            out = np.zeros((len(xs), len(self._ft2col)))

            for ix, conv in tqdm(enumerate(xs), desc='ConvVec: Transforming by conversations', total=len(xs)):
                for k, v in conv_floats(conv).items():
                    out[ix, self._ft2col[k]] = v

                for k, v in conv_ints(conv).items():
                    out[ix, self._ft2col[k]] = v

                if include_ids:
                    ids[conv.convo_id] = ix

            out = self._normalize(out)

            if include_ids:
                return out, ids

            return out
        else:
            raise ValueError


class UserVectorizer(Vectorizer):

    """
    Vectorizer for creating user parameter vectors
    """

    def __init__(self, normalization=None):
        """
        Constructor for UserVectorizer

        Parameters
        ----------
        normalization : None or str
            Can be None, 'minmax', 'mean', or 'standard'
        """
        super(UserVectorizer, self).__init__(normalization)

    def fit(self, xs):
        """
        Fits normalization parameters

        Parameters
        ----------
        xs : Conversation, List(Conversation), List(UniMessage)

        Returns
        -------
        UserVectorizer
        """
        if type(xs) == list:
            if isinstance(xs[0], Conversation):
                values = None

                # compute total users
                seen_user = set()
                for conv in xs:
                    for pid in conv.posts:
                        author = conv.posts[pid].author
                        if author in seen_user:
                            continue

                        seen_user.add(author)
                total_users = len(seen_user)

                # actual fit loop
                seen_user = set()
                ix = 0
                for conv in tqdm(xs, desc='UserVec: Fitting by conversations', total=len(xs)):
                    for pid in conv.posts:
                        author = conv.posts[pid].author
                        if author in seen_user:
                            continue

                        params = {**uac_floats(author, xs), **uac_ints(author, xs)}
                        if values is None:
                            values = np.zeros((total_users, len(params)))
                            self._ft2col = {}
                            # self._num_bool = len(user_bools(user, conv))
                            for k in params:
                                self._ft2col[k] = len(self._ft2col)

                        for k, v in params.items():
                            values[ix, self._ft2col[k]] = v

                        seen_user.add(author)
                        ix += 1

                self._fit_params(values)

                return self
            elif isinstance(xs[0], UniMessage):
                x_ = Conversation(posts={post.uid: post for post in xs})
                return self.fit(x_)
        elif isinstance(xs, Conversation):
            values = None
            total_users = len(messages_per_user(xs))
            for ix, user in tqdm(enumerate(iter_over_users(xs)), desc='fitting users', total=total_users):
                params = {**user_floats(user, xs), **user_ints(user, xs)}

                if values is None:
                    values = np.zeros((total_users, len(params)))
                    self._ft2col = {}
                    self._num_bool = len(user_bools(user, xs))
                    for k in params:
                        self._ft2col[k] = len(self._ft2col)

                for k, v in params.items():
                    values[ix, self._ft2col[k]] = v

            self._fit_params(values)

            return self

        raise ValueError()

    def transform(self, xs, include_ids=False):
        """
        Returns a set of user vectors for each unique user found

        Parameters
        ----------
        xs : Conversation,  List(Conversation), or List(UniMessage)
        include_ids : bool

        Returns
        -------
        dict(str, np.array)
        """
        ids = {}
        if type(xs) == list:
            if isinstance(xs[0], Conversation):
                # compute total users
                seen_user = set()
                for conv in xs:
                    for pid in conv.posts:
                        author = conv.posts[pid].author
                        if author in seen_user:
                            continue

                        seen_user.add(author)
                total_users = len(seen_user)

                out = np.zeros((total_users, len(self._ft2col)))
                # out_bools = np.zeros((total_users, self._num_bool))

                seen_user = set()
                ix = 0
                for conv in tqdm(xs, desc='UserVec: Transforming by conversations', total=len(xs)):
                    for pid in conv.posts:
                        author = conv.posts[pid].author
                        if author in seen_user:
                            continue

                        for k, v in {**uac_ints(author, xs), **uac_floats(author, xs)}.items():
                            out[ix, self._ft2col[k]] = v

                        if include_ids:
                            ids[author] = ix

                        seen_user.add(author)
                        ix += 1

                    # for ik, (_, v) in enumerate(sorted(user_bools(user, conv).items(), key=lambda kv: kv[0])):
                    #     out_bools[ix, ik] = 1 if v else 0

                out = self._normalize(out)
                # out = np.hstack((out, out_bools))

                if include_ids:
                    return out, ids

                return out
            elif isinstance(xs[0], UniMessage):
                x_ = Conversation(posts={post.uid: post for post in xs})
                return self.transform(x_, include_ids=include_ids)
        elif isinstance(xs, Conversation):
            total_users = len(messages_per_user(xs))
            out = np.zeros((total_users, len(self._ft2col)))
            out_bools = np.zeros((total_users, self._num_bool))

            for ix, user in tqdm(enumerate(iter_over_users(xs)), desc='UserVec: Transforming users by user', total=total_users):
                for k, v in {**user_ints(user, xs), **user_floats(user, xs)}.items():
                    out[ix, self._ft2col[k]] = v

                for ik, (_, v) in enumerate(sorted(user_bools(user, xs).items(), key=lambda kv: kv[0])):
                    out_bools[ix, ik] = 1 if v else 0

                if include_ids:
                    ids[user] = ix

            out = self._normalize(out)
            out = np.hstack((out, out_bools))

            if include_ids:
                return out, ids

            return out

        raise ValueError
