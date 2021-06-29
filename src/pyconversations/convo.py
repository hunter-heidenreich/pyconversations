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
        return self._posts

    @property
    def edges(self):
        return self._edges

    def add_post(self, post):
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
        # remove from post dictionary
        del self._posts[uid]

        # update knowledge of edges
        del self._edges[uid]

        # clear cached stats
        self._stats = {}

    def __add__(self, other):
        convo = Conversation()
        for post in other.posts.values():
            convo.add_post(post)
        for post in self.posts.values():
            convo.add_post(post)
        return convo

    def _build_graph(self, directed=False):
        if not directed:
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
        else:
            if 'digraph' in self._cache:
                return self._cache['digraph']

            graph = nx.DiGraph()

            # add posts as nodes
            for uid in self._posts:
                graph.add_node(uid)

            # add reply connections as sedges
            for uid, reps in self._edges.items():
                for rid in reps:
                    if uid in self._posts and rid in self._posts:
                        graph.add_edge(rid, uid)
            self._cache['digraph'] = graph
            return graph

    def segment(self):
        """
        Analyze current post and edge configuration,
        generating sub-conversations
        if there are disparate components
        :return:
        """
        segments = []
        for node_set in nx.connected_components(self._build_graph()):
            convo = Conversation()
            for uid in node_set:
                convo.add_post(self.posts[uid])
            segments.append(convo)

        return segments

    def to_json(self):
        return [post.to_json() for post in self.posts.values()]

    @staticmethod
    def from_json(raw, cons):
        convo = Conversation()
        for p in [cons.from_json(pjson) for pjson in raw]:
            convo.add_post(p)
        return convo

    @property
    def messages(self):
        try:
            return self._stats['messages']
        except KeyError:
            self._stats['messages'] = len(self._posts)
            return self._stats['messages']

    @property
    def connections(self):
        try:
            return self._stats['connections']
        except KeyError:
            self._stats['connections'] = sum(map(lambda x: len({r for r in x if r in self._posts}), self._edges.values()))
            return self._stats['connections']

    @property
    def users(self):
        try:
            return self._stats['users']
        except KeyError:
            self._stats['users'] = len(set([post.author for post in self._posts.values()]))
            return self._stats['users']

    @property
    def chars(self):
        try:
            return self._stats['chars']
        except KeyError:
            self._stats['chars'] = sum(map(lambda x: x.chars, self._posts.values()))
            return self._stats['chars']

    @property
    def tokens(self):
        try:
            return self._stats['tokens']
        except KeyError:
            self._stats['tokens'] = sum(map(lambda x: len(x.tokens), self._posts.values()))

            return self._stats['tokens']

    @property
    def token_types(self):
        try:
            return self._stats['token_types']
        except KeyError:
            self._stats['token_types'] = len(set(
                reduce(lambda x, y: x | y, map(lambda x: x.types, self._posts.values()))))
            return self._stats['token_types']

    @property
    def sources(self):
        # any posts that don't have a predecessor within the conversation
        try:
            return self._stats['sources']
        except KeyError:
            es = {uid: set([e for e in ex if e in self._posts]) for uid, ex in self._edges.items()}
            self._stats['sources'] = {uid for uid in es if not es[uid]}
            return self._stats['sources']

    @property
    def density(self):
        return nx.density(self._build_graph())

    @property
    def degree_hist(self):
        return nx.degree_histogram(self._build_graph())

    @property
    def replies(self):
        if 'replies' not in self._stats:
            rep_cnts = defaultdict(int)
            for post in self._posts.values():
                for rid in post.reply_to:
                    rep_cnts[rid] += 1
            self._stats['replies'] = rep_cnts

        return self._stats['replies']

    @property
    def reply_counts(self):
        # for each post, we'll have a 3-tuple of form (total replies, replies in, replies out)
        reps = self.replies
        total = sum(reps.values())
        return [(total, reps[pid], len(self.posts[pid].reply_to)) for pid in self.posts]

    @property
    def in_degree_hist(self):
        rep_cnts = self.replies
        return [rep_cnts[pid] for pid in self.posts]

    @property
    def out_degree_hist(self):
        return list(self.replies.values())

    def get_depth(self, uid):
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
        if 'depths' in self._stats:
            return self._stats['depths']

        self._stats['depths'] = [self.get_depth(uid) for uid in self.posts]

        return self._stats['depths']

    @property
    def tree_depth(self):
        if 'tree_depth' not in self._stats:
            self._stats['tree_depth'] = max(self.depths)

        return self._stats['tree_depth']

    @property
    def widths(self):
        if 'widths' not in self._stats:
            cnts = dict(Counter(self.depths))
            self._stats['widths'] = [cnts.get(ix, 0) for ix in range(self.tree_depth + 1)]

        return self._stats['widths']

    @property
    def tree_width(self):
        if 'tree_width' not in self._stats:
            self._stats['tree_width'] = max(self.widths)

        return self._stats['tree_width']

    def filter(self, by_langs=None, min_chars=1, before=None, after=None, by_tags=None):
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
        if self.time_order:
            return [self._posts[uid].text for uid in self.time_order]
        else:
            return [self._posts[uid].text for uid in self._posts]

    @property
    def start_time(self):
        try:
            return self._stats['start_time']
        except KeyError:
            self._stats['start_time'] = self._posts[self.time_order[0]].created_at if self.time_order else None
            return self._stats['start_time']

    @property
    def end_time(self):
        try:
            return self._stats['end_time']
        except KeyError:
            self._stats['end_time'] = self._posts[self.time_order[-1]].created_at if self.time_order else None
            return self._stats['end_time']

    @property
    def duration(self):
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
        if self.time_order:
            return [self._posts[uid].created_at.timestamp() for uid in self.time_order]

        return None

    def redact(self):
        rd = {}
        for uid in self._posts:
            for user in self._posts[uid].get_mentions():
                if user not in rd:
                    rd[user] = f'USER{len(rd)}'

        for uid in self._posts:
            self._posts[uid].redact(rd)
