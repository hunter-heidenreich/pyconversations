from collections import defaultdict

import networkx as nx

from .message import get_constructor_by_platform


class Conversation:
    """
    A container class for managing collections of UniMessage (post) objects.
    """

    def __init__(self, posts=None, convo_id=None):
        """
        Constructor for Conversation object.

        Parameters
        ---------
        posts
            An optional dictionary of messages/posts; keys should be unique IDs.
        """
        if not posts:
            posts = {}

        self._posts = posts  # uid -> post object
        self._convo_id = convo_id

        self._relation_map = defaultdict(dict)
        self._author_set = set()

    def __add__(self, other):
        """
        Defines the addition operation over Conversation objects.
        Returns a new copy of a conversation.

        Parameters
        ---------
        other : UniMessage
            Another conversation to be added to this one.

        Returns
        -------
        Conversation
            The combination of this conversation and the conversation in `other`
        """

        convo = Conversation(convo_id=self.convo_id + '++' + other.convo_id)
        for post in other.posts.values():
            convo.add_post(post)
        for post in self.posts.values():
            convo.add_post(post)
        return convo

    @property
    def posts(self):
        """
        Returns a dictionary of posts, keyed by their UIDs.

        Returns
        -------
        dict(UID, UniMessage)
            The dictionary of posts contained in this Conversation object
        """
        return self._posts

    @property
    def convo_id(self):
        """
        The conversation identifier

        Returns
        -------
        Any (or str)
            Returns a conversation identifier. Creates ones from sources if unspecified.
        """
        return self._convo_id if self._convo_id else 'CONV_' + '-'.join(map(str, sorted(self.get_sources())))

    @property
    def authors(self):
        if not self._author_set:
            self._author_set = set([self.posts[pid].author for pid in self.posts])

        return self._author_set

    def add_post(self, post):
        """
        Adds a post to the conversational container.

        Parameters
        ---------
        post : UniMessage, or derivative concrete class
            The post object to be added.

        Returns
        -------
        None
        """
        if post.uid in self._posts and self._posts[post.uid]:
            self._posts[post.uid] |= post
        else:
            self._posts[post.uid] = post

        self._author_set.add(post.author)

    def remove_post(self, uid):
        """
        Deletes a post from the conversational container using its UID.

        Parameters
        ---------
        uid : Hashable
            Unique identifier for the post to delete.

        Returns
        -------
        None
        """
        del self._posts[uid]
        self._author_set = set()

    def as_graph(self):
        """
        Constructs (and returns) a networkx Graph object
        from the contained posts and edges.

        Returns
        -------
        networkx.Graph
            The networkx graph associated with this Conversation
        """
        graph = nx.Graph()

        # add posts as nodes
        for uid in self._posts:
            graph.add_node(uid)

        # add reply connections as edges
        for uid, post in self._posts.items():
            for rid in post.reply_to:
                if uid in self._posts and rid in self._posts:
                    graph.add_edge(uid, rid)

        return graph

    def segment(self):
        """
        Segments a conversation into disjoint (i.e., not connected by any replies) sub-conversations.
        If a single conversation is contained in this object,
        this function will return a list with a single element: a copy of this object.

        Returns
        -------
        list(Conversation)
            A list of sub-conversations
        """
        segments = []
        for node_set in nx.connected_components(self.as_graph()):
            convo = Conversation()
            for uid in node_set:
                convo.add_post(self.posts[uid])
            segments.append(convo)

        return segments

    def to_json(self):
        """
        Returns a JSON representation of this object.

        Returns
        -------
        list(JSON/dict)
            The dictionary/JSON representation of the Conversation
        """
        return [post.to_json() for post in self.posts.values()]

    @staticmethod
    def from_json(raw):
        """
        Converts a JSON representation of a Conversation into a full object.

        Parameters
        ---------
        raw : JSON/dict
            The raw JSON

        Returns
        -------
        Conversation
            The conversation read from the raw JSON
        """
        convo = Conversation()
        for p in [get_constructor_by_platform(pjson['platform']).from_json(pjson) for pjson in raw]:
            convo.add_post(p)
        return convo

    def get_sources(self):
        """
        Returns the originating (non-reply) posts included in this conversation.

        Returns
        -------
        set(UID)
            The set of unique IDs of posts that originate conversation (are not replies)
        """
        return {uid for uid, post in self._posts.items() if not {rid for rid in post.reply_to if rid in self._posts}}

    def filter(self, by_langs=None, min_chars=0, before=None, after=None, by_tags=None, by_platform=None, by_author=None):
        """
        Returns the set of post UIDs that meet the parameterized criteria

        Parameters
        ---------
        by_langs : set(str)
            The desired language codes to be retained. (Default: None)
        min_chars : int
            The minimum number of characters a post should have. (Default: 0)
        before : datetime.datetime
            The earliest datetime desired. (Default: None)
        after : datetime.datetime
            The latest datetime desired. (Default: None)
        by_tags : set(str)
            The required tags. (Default: None)
        by_platform : set(str)
            A set of string names of platforms that should be retained
        by_author : str
            An author

        Returns
        -------
        set(hashable)
            Set of UIDs
        """
        drop = set()
        keep = set(self.posts.keys())
        for uid, post in self._posts.items():
            if by_author is not None and post.author != by_author:
                drop.add(uid)
                continue

            if len(post.text) < min_chars:
                drop.add(uid)
                continue

            if by_langs and post.lang not in by_langs:
                drop.add(uid)
                continue

            if before and (post.created_at is None or post.created_at >= before):
                drop.add(uid)
                continue
            if after and (post.created_at is None or post.created_at <= after):
                drop.add(uid)
                continue

            if by_tags and by_tags != (by_tags & post.tags):
                drop.add(uid)
                continue

            if by_platform and post.platform not in by_platform:
                drop.add(uid)
                continue

        keep -= drop
        return keep

    def time_order(self):
        """
        Returns a time series of the UIDs of posts within this Conversation.

        Returns
        -------
        list(UID)
            The list of UIDs of the posts in the conversation, in temporal order
        """
        try:
            return sorted(self._posts.keys(), key=lambda k: self._posts[k].created_at)
        except TypeError:
            return []

    def text_stream(self):
        """
        Returns the text of the Conversation as a single stream.
        If timestamps are available, text will appear in temporal order.

        Returns
        -------
        list(str)
            The text of the conversation, by post, in temporal order (if available)
        """
        order = self.time_order()
        if order:
            return [self._posts[uid].text for uid in order]
        else:
            return [self._posts[uid].text for uid in self._posts]

    def redact(self, assign_ints=True):
        """
        Redacts user information from the conversation.

        Parameters
        ----------
        assign_ints : bool
            If True, assigns a unique integer to each user such the user will be referred to as `USER><d+>`
            Otherwise, all user redactions will become a `USER` token.

        Returns
        -------
        None
        """
        rd = {}
        for uid in self._posts:
            for user in self._posts[uid].get_mentions():
                if user not in rd:
                    rd[user] = f'USER{len(rd)}' if assign_ints else 'USER'

        for uid in self._posts:
            self._posts[uid].redact(rd)

    def get_ancestors(self, uid, include_post=False):
        """
        Returns the ancestor posts/path for post `uid`.

        Parameters
        ----------
        uid : Hashable
            The unique identifier of desired post

        include_post : bool
            Whether the post should be included in returned collection. Default: False

        Returns
        -------
        Conversation
            The collection of ancestor posts
        """
        if 'ancestors' in self._relation_map and uid in self._relation_map['ancestors']:
            pids = self._relation_map['ancestors'][uid]
            filt_ps = {pid: self.posts[pid] for pid in pids}
        else:
            # get parents
            ps = self.get_parents(uid)
            filt_ps = dict(ps.posts)

            # for each parent, add its ancestors
            for pid in ps.posts:
                for xid in self.get_ancestors(pid).posts:
                    if xid not in filt_ps:
                        filt_ps[xid] = self.posts[xid]

            self._relation_map['ancestors'][uid] = set(filt_ps.keys())

        ancestors = Conversation(posts=filt_ps, convo_id=self.convo_id + '-' + str(uid) + '-ancestors')

        if include_post:
            ancestors.add_post(self.posts[uid])

        return ancestors

    def get_descendants(self, uid, include_post=False):
        """
        Returns the descendant sub-tree for post `uid`.

        Parameters
        ----------
        uid : Hashable
            The unique identifier of desired post

        include_post : bool
            Whether the post should be included in returned collection. Default: False

        Returns
        -------
        Conversation
            The collection of descendant posts
        """
        if 'descendant' in self._relation_map and uid in self._relation_map['descendant']:
            pids = self._relation_map['descendant'][uid]
            filt_ps = {pid: self.posts[pid] for pid in pids}
        else:
            # get children
            ps = self.get_children(uid)
            filt_ps = dict(ps.posts)

            # for each child, add its descendants
            for pid in ps.posts:
                for xid in self.get_descendants(pid).posts:
                    if xid not in filt_ps:
                        filt_ps[xid] = self.posts[xid]

            self._relation_map['descendant'][uid] = set(filt_ps.keys())

        descendants = Conversation(posts=filt_ps, convo_id=self.convo_id + '-' + str(uid) + '-descendant')

        if include_post:
            descendants.add_post(self.posts[uid])

        return descendants

    def get_parents(self, uid, include_post=False):
        """
        Returns the parent(s) of a post specified by `uid`.

        Parameters
        ----------
        uid : Hashable
            The unique identifier of desired post

        include_post : bool
            Whether the post should be included in returned collection. Default: False

        Returns
        -------
        Conversation
            The collection of parent posts
        """
        if 'parents' in self._relation_map and uid in self._relation_map['parents']:
            pids = self._relation_map['parents'][uid]
            filt_ps = {pid: self.posts[pid] for pid in pids}
        else:
            filt_ps = {pid: post for pid, post in self.posts.items() if pid in self.posts[uid].reply_to}
            self._relation_map['parents'][uid] = set(filt_ps.keys())

        cx = Conversation(posts=filt_ps, convo_id=self.convo_id + '-' + str(uid) + '-parents')

        if include_post:
            cx.add_post(self.posts[uid])

        return cx

    def get_children(self, uid, include_post=False):
        """
        Returns the children of a post specified by `uid`.

        Parameters
        ----------
        uid : Hashable
            The unique identifier of desired post

        include_post : bool
            Whether the post should be included in returned collection. Default: False

        Returns
        -------
        Conversation
            The collection of children posts
        """
        if 'children' in self._relation_map and uid in self._relation_map['children']:
            pids = self._relation_map['children'][uid]
            filt_ps = {pid: self.posts[pid] for pid in pids}
        else:
            filt_ps = {pid: post for pid, post in self.posts.items() if uid in self.posts[pid].reply_to}
            self._relation_map['children'][uid] = set(filt_ps.keys())

        cx = Conversation(posts=filt_ps, convo_id=self.convo_id + '-' + str(uid) + '-children')

        if include_post:
            cx.add_post(self.posts[uid])

        return cx

    def get_siblings(self, uid, include_post=False):
        """
        Returns the siblings of a post specified by `uid`.
        Siblings are the child posts of this post's parent posts.

        Parameters
        ----------
        uid : Hashable
            The unique identifier of desired post

        include_post : bool
            Whether the post should be included in returned collection. Default: False

        Returns
        -------
        Conversation
            The collection of sibling posts
        """
        if 'siblings' in self._relation_map and uid in self._relation_map['siblings']:
            pids = self._relation_map['siblings'][uid]
            siblings = Conversation(posts={pid: self.posts[pid] for pid in pids}, convo_id=self.convo_id + '-' + str(uid) + '-siblings')
        else:
            # just caches the parent IDs
            self.get_parents(uid)

            siblings = Conversation(convo_id=self.convo_id + '-' + str(uid) + '-siblings')
            for pid in self._relation_map['parents'][uid]:
                siblings += self.get_children(pid)

            self._relation_map['siblings'][uid] = set(siblings.posts.keys())

        if uid in siblings.posts and not include_post:
            siblings.remove_post(uid)

        if include_post and uid not in siblings.posts:
            siblings.add_post(self.posts[uid])

        return siblings

    def get_before(self, uid, include_post=False):
        """
        Returns the collection of posts in this conversation that were created before the post
        with UID `uid`

        Parameters
        ----------
        uid : Hashable
            The UID of the post that is the pivot

        include_post : bool
            Whether the post should be included in returned collection. Default: False

        Returns
        -------
        Conversation
            The collection of posts posted before uid

        Raises
        ------
        KeyError
            When `uid` is not in the Conversation
        """
        if 'before' in self._relation_map and uid in self._relation_map['before']:
            pids = self._relation_map['before'][uid]
            filt_posts = {pid: self.posts[pid] for pid in pids}
        else:
            filt_posts = {pid: self.posts[pid] for pid in self.filter(before=self._posts[uid].created_at)}
            self._relation_map['before'][uid] = set(filt_posts.keys())

        cx = Conversation(posts=filt_posts, convo_id=self.convo_id + '-' + str(uid) + '-before')

        if include_post:
            cx.add_post(self.posts[uid])

        return cx

    def get_after(self, uid, include_post=False):
        """
        Returns the collection of posts in this conversation that were created after the post
        with UID `uid`

        Parameters
        ----------
        uid : Hashable
            The UID of the post that is the pivot

        include_post : bool
            Whether the post should be included in returned collection. Default: False

        Returns
        -------
        Conversation
            The collection of posts posted after uid

        Raises
        ------
        KeyError
            When `uid` is not in the Conversation
        """
        if 'after' in self._relation_map and uid in self._relation_map['after']:
            pids = self._relation_map['after'][uid]
            filt_posts = {pid: self.posts[pid] for pid in pids}
        else:
            filt_posts = {pid: self.posts[pid] for pid in self.filter(after=self._posts[uid].created_at)}
            self._relation_map['after'][uid] = set(filt_posts.keys())

        cx = Conversation(posts=filt_posts, convo_id=self.convo_id + '-' + str(uid) + '-after')

        if include_post:
            cx.add_post(self.posts[uid])

        return cx
