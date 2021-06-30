from collections import Counter
from collections import defaultdict
from functools import reduce

import networkx as nx


class Conversation:
    """A conversational container for the PyConversations package."""

    def __init__(self, posts=None, edges=None):
        """
        Constructor for Conversation object.

        Parameters
        ---------
        posts
            An optional dictionary of messages/posts; keys should be unique IDs.
        edges
            An optional dictionary containing information of the posts replied to, also indexed by unique IDs.
        """
        if not posts:
            posts = {}

        if not edges:
            edges = {}

        self._posts = posts  # uid -> post object
        self._edges = edges  # uid -> {reply_tos}
        self._stats = {}
        self._cache = {}

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
    def edges(self):
        """
        Returns a dictionary of reply edges, keyed by post UIDs.

        Returns
        -------
        dict(UID, set(UID))
            The dictionary reply "edges" that map a post to the UIDs of the posts it replies to
        """
        return self._edges

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

            # update knowledge of edges
            self._edges[post.uid] |= post.reply_to
        else:
            # to dictionary
            self._posts[post.uid] = post

            # update knowledge of edges
            self._edges[post.uid] = post.reply_to

        # clear cached stats
        self._stats = {}

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
        # remove from post dictionary
        del self._posts[uid]

        # update knowledge of edges
        del self._edges[uid]

        # clear cached stats
        self._stats = {}

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

        convo = Conversation()
        for post in other.posts.values():
            convo.add_post(post)
        for post in self.posts.values():
            convo.add_post(post)
        return convo

    def _build_graph(self):
        """
        Constructs (and returns) a networkx Graph object
        from the contained posts and edges.

        Returns
        -------
        networkx.Graph
            The networkx graph associated with this Conversation
        """
        if 'graph' in self._cache:
            return self._cache['graph']

        graph = nx.Graph()

        # add posts as nodes
        for uid in self._posts:
            graph.add_node(uid)

        # add reply connections as sedges
        for uid, reps in self._edges.items():
            for rid in reps:
                if uid in self._posts and rid in self._posts:
                    graph.add_edge(uid, rid)
        self._cache['graph'] = graph
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
        for node_set in nx.connected_components(self._build_graph()):
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
    def from_json(raw, cons):
        """
        Converts a JSON representation of a Conversation into a full object.

        Parameters
        ---------
        raw : JSON/dict
            The raw JSON
        cons : UniMessage constructor
            The post/UniversalMessage constructor to use.

        Returns
        -------
        Conversation
            The conversation read from the raw JSON
        """
        convo = Conversation()
        for p in [cons.from_json(pjson) for pjson in raw]:
            convo.add_post(p)
        return convo

    @property
    def messages(self):
        """
        Returns the number of messages contained in this conversation as an integer.

        Returns
        -------
        int
            Number of messages in the conversation
        """
        try:
            return self._stats['messages']
        except KeyError:
            self._stats['messages'] = len(self._posts)
            return self._stats['messages']

    @property
    def connections(self):
        """
        Returns the number of reply connections contained in this conversation as an integer.

        Returns
        -------
        int
            Number of reply connections in conversation
        """
        try:
            return self._stats['connections']
        except KeyError:
            self._stats['connections'] = sum(map(lambda x: len({r for r in x if r in self._posts}), self._edges.values()))
            return self._stats['connections']

    @property
    def users(self):
        """
        Returns the number of unique users participating in a conversations as an integer.

        Returns
        -------
        int
            Number of unique users participating in the conversation
        """
        try:
            return self._stats['users']
        except KeyError:
            self._stats['users'] = len(set([post.author for post in self._posts.values()]))
            return self._stats['users']

    @property
    def chars(self):
        """
        Returns the integer character length of the entire conversation.
        This is a summation over the character counts for all posts within it.

        Returns
        -------
        int
            Length of the conversation in characters
        """
        try:
            return self._stats['chars']
        except KeyError:
            self._stats['chars'] = sum(map(lambda x: x.chars, self._posts.values()))
            return self._stats['chars']

    @property
    def tokens(self):
        """
        Returns the integer token length of the entire conversation.
        This is a summation over the token counts for all posts within it.

        Returns
        -------
        int
            Length of the conversation in tokens
        """
        try:
            return self._stats['tokens']
        except KeyError:
            self._stats['tokens'] = sum(map(lambda x: len(x.tokens), self._posts.values()))

            return self._stats['tokens']

    @property
    def token_types(self):
        """
        Returns the number of unique tokens used in this conversation (as an integer).

        Returns
        -------
        int
            Size of the conversation vocabulary
        """
        try:
            return self._stats['token_types']
        except KeyError:
            self._stats['token_types'] = len(set(
                reduce(lambda x, y: x | y, map(lambda x: x.types, self._posts.values()))))
            return self._stats['token_types']

    @property
    def sources(self):
        """
        Returns the originating (non-reply) posts included in this conversation.

        Returns
        -------
        set(UID)
            The set of unique IDs of posts that originate conversation (are not replies)
        """
        try:
            return self._stats['sources']
        except KeyError:
            es = {uid: set([e for e in ex if e in self._posts]) for uid, ex in self._edges.items()}
            self._stats['sources'] = {uid for uid in es if not es[uid]}
            return self._stats['sources']

    @property
    def density(self):
        """
        Returns the density (a float) of the conversation,
        when represented as a graph.

        Returns
        -------
        float
            The density of the conversation graph
        """
        return nx.density(self._build_graph())

    @property
    def degree_hist(self):
        """
        Returns the degree (# of replies received) histogram of this conversation.

        Returns
        -------
        list(int)
            A list of frequencies of degrees.
            The degree values are the index in the list.
        """
        return nx.degree_histogram(self._build_graph())

    @property
    def replies(self):
        """
        Returns the number of replies received (as collected in this Conversation)
        for each post within the Conversation.

        Returns
        -------
        dict(UID, int)
            Mapping from post UID to number of replies received
        """
        if 'replies' not in self._stats:
            rep_cnts = defaultdict(int)
            for post in self._posts.values():
                for rid in post.reply_to:
                    rep_cnts[rid] += 1
            self._stats['replies'] = rep_cnts

        return self._stats['replies']

    @property
    def reply_counts(self):
        """
        Returns a list of 3-tuples of the form (total replies, replies in, replies out) for each post

        Returns
        -------
        list(3-tuple(total replies in conversation, replies received, replies out))
            List of 3-tuples of the form (total replies, replies in, replies out) for each post
        """
        # for each post, we'll have a 3-tuple of form (total replies, replies in, replies out)
        reps = self.replies
        total = sum(reps.values())
        return [(total, reps[pid], len(self.posts[pid].reply_to)) for pid in self.posts]

    @property
    def in_degree_hist(self):
        """
        Returns a list of all in-degrees.

        Returns
        -------
        list(int)
            List of the replies received for each post
        """
        rep_cnts = self.replies
        return [rep_cnts[pid] for pid in self.posts]

    @property
    def out_degree_hist(self):
        """
        Returns a list of all out-degrees.

        Returns
        -------
        list(int)
            List of the replies generated for each post
        """
        return list(self.replies.values())

    def get_depth(self, uid):
        """
        Returns the depth of a specific post within this Conversation.

        Parameters
        ---------
        uid : Hashable
            The unique identifier of the post

        Returns
        -------
        int
            The depth of the post
        """
        if 'depth' not in self._stats:
            self._stats['depth'] = {}

        if uid not in self._stats['depth']:
            if self._posts[uid].reply_to:
                reply = self._posts[uid]
                depth = None

                for rid in self._posts[uid].reply_to:
                    if rid in self._posts:
                        post = self._posts[rid]

                        if (reply.created_at and post.created_at and reply.created_at > post.created_at) or \
                           post.created_at is None or reply.created_at is None:

                            d = self.get_depth(rid) + 1
                            if depth is None or d < depth:
                                depth = d

                if depth is None:
                    depth = 0
            else:
                depth = 0

            self._stats['depth'][uid] = depth

        return self._stats['depth'][uid]

    @property
    def depths(self):
        """
        Returns a list of depths of posts within this Conversation.
        This is useful for understanding how the Conversation is structured/dispersed.

        Returns
        -------
        list(int)
            List of the depths of each post
        """
        if 'depths' in self._stats:
            return self._stats['depths']

        self._stats['depths'] = [self.get_depth(uid) for uid in self.posts]

        return self._stats['depths']

    @property
    def tree_depth(self):
        """
        Returns the depth of this Conversation.
        Specifically, the longest path from source to leaf.

        Returns
        -------
        int
            Depth of the conversation DAG
        """
        if 'tree_depth' not in self._stats:
            self._stats['tree_depth'] = max(self.depths)

        return self._stats['tree_depth']

    @property
    def widths(self):
        """
        Returns a list of the width (# of posts) at each depth level within the Conversation.

        Returns
        -------
        list(int)
            List of the width (# of posts) of each depth level
        """
        if 'widths' not in self._stats:
            cnts = dict(Counter(self.depths))
            self._stats['widths'] = [cnts.get(ix, 0) for ix in range(self.tree_depth + 1)]

        return self._stats['widths']

    @property
    def tree_width(self):
        """
        Returns the width of the full Conversation (the max width of any depth level).

        Returns
        -------
        int
            Width of the tree

        Notes
        -----
        The width of the conversation is equal to the fattest depth level.
        """
        if 'tree_width' not in self._stats:
            self._stats['tree_width'] = max(self.widths)

        return self._stats['tree_width']

    def filter(self, by_langs=None, min_chars=1, before=None, after=None, by_tags=None):
        """
        Removes posts from this Conversation based on specified parameters.

        Parameters
        ---------
        by_langs : set(str)
            The desired language codes to be retained. (Default: None)
        min_chars : int
            The minimum number of characters a post should have. (Default: 1)
        before : datetime.datetime
            The earliest datetime desired. (Default: None)
        after : datetime.datetime
            The latest datetime desired. (Default: None)
        by_tags : set(str)
            The required tags. (Default: None)

        Returns
        -------
        None
        """
        drop = set()
        for uid, post in self._posts.items():
            if len(post.text) < min_chars:
                drop.add(uid)
                continue

            if by_langs and post.lang not in by_langs:
                drop.add(uid)
                continue

            if before and (post.created_at is None or post.created_at > before):
                drop.add(uid)
                continue
            if after and (post.created_at is None or post.created_at < after):
                drop.add(uid)
                continue

            if by_tags and by_tags != (by_tags & post.tags):
                drop.add(uid)
                continue

        for uid in drop:
            self.remove_post(uid)

    @property
    def time_order(self):
        """
        Returns a time series of the UIDs of posts within this Conversation.

        Returns
        -------
        list(UID)
            The list of UIDs of the posts in the conversation, in temporal order
        """
        try:
            return self._stats['time_order']
        except KeyError:
            try:
                self._stats['time_order'] = sorted(self._posts.keys(), key=lambda k: self._posts[k].created_at)
            except TypeError:
                self._stats['time_order'] = None
            return self._stats['time_order']

    @property
    def text_stream(self):
        """
        Returns the text of the Conversation as a single stream.
        If timestamps are available, text will appear in temporal order.

        Returns
        -------
        list(str)
            The text of the conversation, by post, in temporal order (if available)
        """
        if self.time_order:
            return [self._posts[uid].text for uid in self.time_order]
        else:
            return [self._posts[uid].text for uid in self._posts]

    @property
    def start_time(self):
        """
        Returns the start datetime of the Conversation.

        Returns
        -------
        datetime.datetime or None
            The earliest post in the Conversation, if available
        """
        try:
            return self._stats['start_time']
        except KeyError:
            self._stats['start_time'] = self._posts[self.time_order[0]].created_at if self.time_order else None
            return self._stats['start_time']

    @property
    def end_time(self):
        """
        Returns the end datetime of the Conversation.

        Returns
        -------
        datetime.datetime or None
            The latest post in the Conversation, if available
        """
        try:
            return self._stats['end_time']
        except KeyError:
            self._stats['end_time'] = self._posts[self.time_order[-1]].created_at if self.time_order else None
            return self._stats['end_time']

    @property
    def duration(self):
        """
        Returns the duration (in seconds) of the Conversation.

        Returns
        -------
        float
            Length of the conversation in seconds
        """
        try:
            return self._stats['duration']
        except KeyError:
            if self.end_time and self.start_time:
                self._stats['duration'] = (self.end_time - self.start_time).total_seconds()
            else:
                self._stats['duration'] = None
            return self._stats['duration']

    @property
    def time_series(self):
        """
        Returns the time series of the conversation as floating point timestamps.

        Returns
        -------
        list(float)
            List of datetime objects of conversation's posts (in order) as floating-point timestamps
        """
        if self.time_order:
            return [self._posts[uid].created_at.timestamp() for uid in self.time_order]

        return None

    def redact(self):
        """
        Redacts user information from the conversation.

        Returns
        -------
        None
        """
        rd = {}
        for uid in self._posts:
            for user in self._posts[uid].get_mentions():
                if user not in rd:
                    rd[user] = f'USER{len(rd)}'

        for uid in self._posts:
            self._posts[uid].redact(rd)
