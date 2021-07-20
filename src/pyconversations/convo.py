from collections import Counter
from collections import defaultdict

import networkx as nx

from .message import get_constructor_by_platform


class Conversation:
    """A conversational container for the PyConversations package."""

    def __init__(self, posts=None, edges=None, convo_id=None):
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

        self._depth_cache = {}

        self._convo_id = convo_id

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

    @property
    def convo_id(self):
        """
        The conversation identifier

        Returns
        -------
        Any (or str)
            Returns a conversation identifier. Creates ones from sources if unspecified.
        """
        return self._convo_id if self._convo_id else 'CONV_' + '-'.join(map(str, sorted(self.sources)))

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
        graph = nx.Graph()

        # add posts as nodes
        for uid in self._posts:
            graph.add_node(uid)

        # add reply connections as sedges
        for uid, reps in self._edges.items():
            for rid in reps:
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

    @property
    def messages(self):
        """
        Returns the number of messages contained in this conversation as an integer.

        Returns
        -------
        int
            Number of messages in the conversation
        """
        return len(self._posts)

    @property
    def connections(self):
        """
        Returns the number of reply connections contained in this conversation as an integer.

        Returns
        -------
        int
            Number of reply connections in conversation
        """
        return sum(map(lambda x: len({r for r in x if r in self._posts}), self._edges.values()))

    @property
    def users(self):
        """
        Returns the number of unique users participating in a conversations as an integer.

        Returns
        -------
        int
            Number of unique users participating in the conversation
        """
        return len(set([post.author for post in self._posts.values()]))

    @property
    def sources(self):
        """
        Returns the originating (non-reply) posts included in this conversation.

        Returns
        -------
        set(UID)
            The set of unique IDs of posts that originate conversation (are not replies)
        """
        es = {uid: set([e for e in ex if e in self._posts]) for uid, ex in self._edges.items()}
        return {uid for uid in es if not es[uid]}

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
        rep_cnts = defaultdict(int)
        for post in self._posts.values():
            for rid in post.reply_to:
                rep_cnts[rid] += 1
        return rep_cnts

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
        if uid not in self._depth_cache:
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

            self._depth_cache[uid] = depth

        return self._depth_cache[uid]

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
        return [self.get_depth(uid) for uid in self.posts]

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
        return max(self.depths)

    @property
    def widths(self):
        """
        Returns a list of the width (# of posts) at each depth level within the Conversation.

        Returns
        -------
        list(int)
            List of the width (# of posts) of each depth level
        """
        cnts = dict(Counter(self.depths))

        return [cnts.get(ix, 0) for ix in range(self.tree_depth + 1)]

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
        return max(self.widths)

    def filter(self, by_langs=None, min_chars=0, before=None, after=None, by_tags=None, by_platform=None):
        """
        Removes posts from this Conversation based on specified parameters.

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

        Returns
        -------
        Conversation
            A conversation with only the posts retained that meet the specified criteria
        """
        drop = set()
        keep = set(self.posts.keys())
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

            if by_platform and post.platform not in by_platform:
                drop.add(uid)
                continue

        keep -= drop
        filt_ps = {pid: post for pid, post in self.posts.items() if pid in keep}
        filt_es = {pid: es & keep for pid, es in self.edges.items() if pid in keep}

        return Conversation(posts=filt_ps, edges=filt_es)

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
            return sorted(self._posts.keys(), key=lambda k: self._posts[k].created_at)
        except TypeError:
            return []

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
