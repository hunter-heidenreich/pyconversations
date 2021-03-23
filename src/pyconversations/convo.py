import re
from collections import Counter
from functools import reduce

import networkx as nx


class Conversation:

    def __init__(self, posts=None, edges=None):
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
        ps = dict(self._posts)
        es = dict(self._edges)

        # merge from other
        ps.update(other.posts)
        es.update(other.edges)

        return Conversation(posts=ps, edges=es)

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
            segments.append(
                Conversation(posts={uid: self._posts[uid] for uid in node_set},
                             edges={uid: self._edges[uid] for uid in node_set})
            )

        return segments

    def to_json(self):
        return {
            'posts': {uid: post.to_json() for uid, post in self._posts.items()},
            'edges': {uid: list(reps) for uid, reps in self._edges.items()}
        }

    @staticmethod
    def from_json(raw, cons):
        ps = {uid: cons.from_json(pjson) for uid, pjson in raw['posts'].items()}
        es = {uid: set(reps) for uid, reps in raw['edges'].items()}
        return Conversation(posts=ps, edges=es)

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
            self._stats['chars'] = sum(map(lambda x: len(x.text), self._posts.values()))
            return self._stats['chars']

    @property
    def tokens(self):
        try:
            return self._stats['tokens']
        except KeyError:
            self._stats['tokens'] = sum(map(lambda x: len(re.split(r'\s+', x.text)), self._posts.values()))
            return self._stats['tokens']

    @property
    def token_types(self):
        try:
            return self._stats['token_types']
        except KeyError:
            self._stats['token_types'] = set(
                reduce(lambda x, y: x | y, map(lambda x: set(re.split(r'\s+', x.text)), self._posts.values())))
            return self._stats['token_types']

    @property
    def token_types_(self):
        try:
            return self._stats['token_types_']
        except KeyError:
            if 'token_types' in self._stats:
                self._stats['token_types_'] = {t.lower() for t in self._stats['token_types']}
            else:
                self._stats['token_types_'] = set(
                    reduce(lambda x, y: x | y, map(lambda x: set(re.split(r'\s+', x.text.lower())), self._posts.values())))
            return self._stats['token_types_']

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
    def in_degree_hist(self):
        digraph = self._build_graph(directed=True)
        return [digraph.in_degree[n] for n in digraph.nodes]

    @property
    def out_degree_hist(self):
        digraph = self._build_graph(directed=True)
        return [digraph.out_degree[n] for n in digraph.nodes]

    @property
    def depths(self):
        srcs = self.sources
        if len(srcs) > 1:
            print(f'Value Error: Too many sources.')
            print(f'{len(srcs)}: {srcs}')
            raise ValueError

        root = list(srcs)[0]
        return [nx.shortest_path_length(self._build_graph(), root, uid) for uid in self.posts]

    @property
    def tree_depth(self):
        return max(self.depths)

    @property
    def widths(self):
        cnts = dict(Counter(self.depths))
        return [cnts.get(ix, 0) for ix in range(self.tree_depth + 1)]

    @property
    def tree_width(self):
        return max(self.widths)

    @property
    def assortativity(self):
        try:
            return nx.degree_pearson_correlation_coefficient(self._build_graph())
        except nx.exception.NetworkXError:
            return None
        except ValueError:
            return None

    @property
    def diameter(self):
        return nx.algorithms.diameter(self._build_graph(), e=self.eccentricity)

    @property
    def radius(self):
        return nx.algorithms.radius(self._build_graph(), e=self.eccentricity)

    @property
    def eccentricity(self):
        if 'eccentricity' not in self._stats:
            self._stats['eccentricity'] = nx.algorithms.eccentricity(self._build_graph())
        return self._stats['eccentricity']

    @property
    def rich_club_coefficient(self):
        try:
            self._stats['rich_club_coefficient'] = nx.algorithms.rich_club_coefficient(self._build_graph())
        except nx.exception.NetworkXError:
            self._stats['rich_club_coefficient'] = None
        except IndexError:
            self._stats['rich_club_coefficient'] = None
        except nx.exception.NetworkXAlgorithmError:
            self._stats['rich_club_coefficient'] = None

        return self._stats['rich_club_coefficient']

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
