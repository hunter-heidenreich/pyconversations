from abc import ABC
from abc import abstractmethod

import numpy as np
from tqdm import tqdm

from ..convo import Conversation
from ..message import UniMessage
from .conv import ConvoFeatures
from .conv import messages_per_user
from .post import PostFeatures
from .post_in_conv import PostInConvoFeatures
from .user_across_conv import UserAcrossConvoFeatures
from .user_in_conv import UserInConvoFeatures


class Vectorizer(ABC):

    """
    Abstract vectorization class.
    Implements normalization.
    """

    def __init__(self, normalization):
        self._stats = {}

        # feature name to column index
        self._num2col = {}
        self._bool2col = {}

        # Can be None, 'minmax', 'mean', or 'standard'
        self._norm = normalization

    @abstractmethod
    def fit(self, *args):
        """
        Abstract method for fitting normalization and vectorization parameters
        to data in `kwargs`

        Parameters
        ----------
        args : list

        Returns
        -------
        Vectorizer
            This object should return itself
        """
        pass

    @abstractmethod
    def transform(self, *args):
        """
        Abstract method for transforming data into a vector (or vectors)

        Parameters
        ----------
        args : list

        Returns
        -------
        np.ndarray
        dict(Hashable, int)
            Mapping from UID to integer row
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

        self._bool_fns = [PostFeatures.bools]
        self._ic_bool_fns = [PostInConvoFeatures.bools]

        self._num_fns = [PostFeatures.floats, PostFeatures.ints]
        self._ic_num_fns = [PostInConvoFeatures.floats, PostInConvoFeatures.ints]

    def fit(self, xs):
        """
        Fits the parameters necessary for normalization and vectorization of posts.

        Parameters
        ----------
        xs : List(UniMessage) or List(Conversation) or Conversation

        Returns
        -------
        PostVectorizer
        """
        self._num2col = {}
        self._bool2col = {}

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
            if not ix:
                for f in self._bool_fns:
                    for k in f(post):
                        self._bool2col[k] = len(self._bool2col)

                for f in self._num_fns:
                    for k in f(post):
                        self._num2col[k] = len(self._num2col)

                values = np.zeros((len(posts), len(self._num2col)))

            for f in self._num_fns:
                for k, v in f(post).items():
                    values[ix, self._num2col[k]] = v

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
        ix = 0
        total_posts = sum(map(lambda c: len(c.posts), convs))
        values = None

        for conv in tqdm(convs, desc='PostVec: Fitting by conversations'):
            for post in conv.posts.values():
                if not ix:
                    for f in self._ic_bool_fns:
                        for k in f(post, conv):
                            self._bool2col[k] = len(self._bool2col)

                    for f in self._ic_num_fns:
                        for k in f(post, conv):
                            self._num2col[k] = len(self._num2col)

                    values = np.zeros((total_posts, len(self._num2col)))

                for f in self._ic_num_fns:
                    for k, v in f(post, conv).items():
                        values[ix, self._num2col[k]] = v

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
        xs : List(UniMessage) or List(Conversation) or Conversation
        include_ids : bool

        Returns
        -------
        np.array
            (N, d), where N is the number of posts and d is the number of features
        dict(Hashable, int)
            Optional. Returned if include_ids=True and creates a map from UID to row in returned array
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
        out = np.zeros((len(posts), len(self._num2col)))
        out_bools = np.zeros((len(posts), len(self._bool2col)))

        for ix, post in tqdm(enumerate(posts), desc='PostVec: Transforming by posts'):
            for f in self._num_fns:
                for k, v in f(post).items():
                    out[ix, self._num2col[k]] = v

            for f in self._bool_fns:
                for k, v in f(post).items():
                    out_bools[ix, self._bool2col[k]] = 1 if v else 0

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
        np.array
        """
        ix = 0
        ids = {}
        total_posts = sum(map(lambda c: len(c.posts), convs))
        out = np.zeros((total_posts, len(self._num2col)))
        out_bools = np.zeros((total_posts, len(self._bool2col)))

        for conv in tqdm(convs, desc='PostVec: Transforming by conversations'):
            for post in conv.posts.values():
                for f in self._ic_num_fns:
                    for k, v in f(post, conv).items():
                        out[ix, self._num2col[k]] = v

                for f in self._ic_bool_fns:
                    for k, v in f(post, conv).items():
                        out[ix, self._bool2col[k]] = 1 if v else 0

                ids[(conv.convo_id, post.uid)] = ix
                ix += 1

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
        self._num_fns = [ConvoFeatures.floats, ConvoFeatures.ints]
        self._bool_fns = []

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
            self._num2col = {}
            self._bool2col = {}

            for ix, conv in tqdm(enumerate(xs), desc='ConvVec: Fitting by conversations', total=len(xs)):
                if not ix:
                    for f in self._bool_fns:
                        for k in f(conv):
                            self._bool2col[k] = len(self._bool2col)

                    for f in self._num_fns:
                        for k in f(conv):
                            self._num2col[k] = len(self._num2col)

                    values = np.zeros((len(xs), len(self._num2col)))

                for f in self._num_fns:
                    for k, v in f(conv).items():
                        values[ix, self._num2col[k]] = v

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
        np.array
        dict(Hashable, int)
            Optional. Returned if include_ids=True and creates a map from UID to row in returned array
        """
        if isinstance(xs, Conversation):
            return self.transform([xs], include_ids=include_ids)
        elif type(xs) == list and isinstance(xs[0], Conversation):
            ids = {}
            out = np.zeros((len(xs), len(self._num2col)))

            for ix, conv in tqdm(enumerate(xs), desc='ConvVec: Transforming by conversations', total=len(xs)):
                for f in self._num_fns:
                    for k, v in f(conv).items():
                        out[ix, self._num2col[k]] = v

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

        self._bool_fns = [UserInConvoFeatures.bools]
        self._ac_bool_fns = [UserAcrossConvoFeatures.bools]

        self._num_fns = [UserInConvoFeatures.ints, UserInConvoFeatures.floats]
        self._ac_num_fns = [UserAcrossConvoFeatures.ints, UserAcrossConvoFeatures.floats]

        self._across = False

    def _get_user_cnt(self, xs):
        # compute total users
        seen_user = set()
        for conv in xs:
            for pid in conv.posts:
                author = conv.posts[pid].author
                if author in seen_user:
                    continue

                seen_user.add(author)
        total_users = len(seen_user)

        return total_users, seen_user

    def fit(self, xs):
        """
        Fits normalization parameters

        Parameters
        ----------
        xs : Conversation or List(Conversation) or List(UniMessage)

        Returns
        -------
        UserVectorizer
        """
        self._num2col = {}
        self._bool2col = {}

        if type(xs) == list:
            if isinstance(xs[0], Conversation):
                values = None
                total_users, users = self._get_user_cnt(xs)
                for ix, user in tqdm(enumerate(users), desc='UserVec: Fitting by user', total=total_users):
                    if not ix:
                        self._across = True

                        for f in self._ac_bool_fns:
                            for k in f(user, xs):
                                self._bool2col[k] = len(self._bool2col)

                        for f in self._ac_num_fns:
                            for k in f(user, xs):
                                self._num2col[k] = len(self._num2col)

                        values = np.zeros((total_users, len(self._num2col)))

                    for f in self._ac_num_fns:
                        for k, v in f(user, xs).items():
                            values[ix, self._num2col[k]] = v

                self._fit_params(values)

                return self
            elif isinstance(xs[0], UniMessage):
                x_ = Conversation(posts={post.uid: post for post in xs})
                return self.fit(x_)
        elif isinstance(xs, Conversation):
            values = None
            total_users = len(messages_per_user(xs))
            users = xs.authors
            for ix, user in tqdm(enumerate(users), desc='UserVec: Fitting by users', total=total_users):
                if not ix:
                    self._across = False
                    for f in self._bool_fns:
                        for k in f(user, xs):
                            self._bool2col[k] = len(self._bool2col)

                    for f in self._num_fns:
                        for k in f(user, xs):
                            self._num2col[k] = len(self._num2col)

                    values = np.zeros((total_users, len(self._num2col)))

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
        np.arrary
        dict(Hashable, int)
            Optional. Returned if include_ids=True and creates a map from UID to row in returned array
        """
        ids = {}
        if type(xs) == list:
            if isinstance(xs[0], Conversation):
                total_users, users = self._get_user_cnt(xs)

                out = np.zeros((total_users, len(self._num2col)))
                # out_bools = np.zeros((total_users, len(self._bool2col)))

                for ix, user in tqdm(enumerate(users), desc='UserVec: Transforming by users', total=total_users):
                    for f in self._ac_num_fns:
                        for k, v in f(user, xs).items():
                            out[ix, self._num2col[k]] = v

                    if include_ids:
                        ids[user] = ix

                out = self._normalize(out)
                # out = np.hstack((out, out_bools))

                if include_ids:
                    return out, ids

                return out
            elif isinstance(xs[0], UniMessage):
                x_ = Conversation(posts={post.uid: post for post in xs})
                return self.transform(x_, include_ids=include_ids)
        elif isinstance(xs, Conversation):
            if self._across:
                return self.transform([xs], include_ids=include_ids)

            total_users = len(messages_per_user(xs))
            users = xs.authors
            out = np.zeros((total_users, len(self._num2col)))
            out_bools = np.zeros((total_users, len(self._bool2col)))

            for ix, user in tqdm(enumerate(users), desc='UserVec: Transforming users by user', total=total_users):
                for f in self._num_fns:
                    for k, v in f(user, xs).items():
                        out[ix, self._num2col[k]] = v

                for f in self._bool_fns:
                    for k, v in f(user, xs).items():
                        out_bools[ix, self._bool2col[k]] = 1 if v else 0

                if include_ids:
                    ids[user] = ix

            out = self._normalize(out)
            out = np.hstack((out, out_bools))

            if include_ids:
                return out, ids

            return out

        raise ValueError
