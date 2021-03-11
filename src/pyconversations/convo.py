import re
from collections import Counter
from functools import reduce

import networkx as nx
from networkx.algorithms import approximation


class Conversation:

    def __init__(self, posts=None, edges=None):
        if not posts:
            posts = {}

        if not edges:
            edges = {}

        self._posts = posts  # uid -> post object
        self._edges = edges  # uid -> {reply_tos}
        self._stats = {}

    @property
    def posts(self):
        return self._posts

    @property
    def edges(self):
        return self._edges

    def add_post(self, post):
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
        else:
            graph = nx.DiGraph()

            # add posts as nodes
            for uid in self._posts:
                graph.add_node(uid)

            # add reply connections as sedges
            for uid, reps in self._edges.items():
                for rid in reps:
                    if uid in self._posts and rid in self._posts:
                        graph.add_edge(rid, uid)

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

    def from_json(self, raw, cons):
        self._posts = {uid: cons.from_json(pjson) for uid, pjson in raw['posts'].items()}
        self._edges = {uid: set(reps) for uid, reps in raw['edges'].items()}

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

    def _network_stats(self):
        graph = self._build_graph()
        digraph = self._build_graph(directed=True)

        self._stats['density'] = nx.density(graph)
        self._stats['degree_hist'] = nx.degree_histogram(graph)
        self._stats['in_degree_hist'] = dict(Counter([digraph.in_degree[n] for n in digraph.nodes]))
        self._stats['out_degree_hist'] = dict(Counter([digraph.out_degree[n] for n in digraph.nodes]))
        self._stats['max_clique'] = len(approximation.max_clique(graph))
        self._stats['avg_cluster'] = approximation.average_clustering(graph)
        self._stats['tree_width'] = approximation.treewidth_min_degree(graph)[0]

        try:
            self._stats['assortativity'] = nx.algorithms.assortativity.degree_pearson_correlation_coefficient(graph)
        except ValueError:
            self._stats['assortativity'] = None

        # centrality metrics
        try:
            self._stats['centrality_eigen'] = nx.algorithms.centrality.eigenvector_centrality_numpy(graph)
        except TypeError:
            self._stats['centrality_eigen'] = None
        self._stats['centrality_katz'] = nx.algorithms.centrality.katz_centrality_numpy(graph)
        self._stats['centrality_closeness'] = nx.algorithms.centrality.closeness_centrality(graph)
        self._stats['centrality_betweeness'] = nx.algorithms.centrality.betweenness_centrality(graph)
        try:
            self._stats['centrality_current_flow_betweeness'] = nx.algorithms.centrality.current_flow_betweenness_centrality(graph)
        except ZeroDivisionError:
            self._stats['centrality_current_flow_betweeness'] = None
        self._stats['centrality_load'] = nx.algorithms.centrality.load_centrality(graph)
        self._stats['centrality_degree'] = nx.algorithms.centrality.degree_centrality(graph)
        self._stats['centrality_info'] = nx.algorithms.centrality.information_centrality(graph)
        self._stats['centrality_harmonic'] = nx.algorithms.centrality.harmonic_centrality(graph)
        self._stats['centrality_second_order'] = nx.algorithms.centrality.second_order_centrality(graph)

        self._stats['estrada_index'] = nx.algorithms.centrality.estrada_index(graph)

        self._stats['longest_path'] = len(nx.algorithms.dag_longest_path(digraph))

        self._stats['eccentricity'] = nx.algorithms.eccentricity(graph)
        self._stats['diameter'] = nx.algorithms.diameter(graph, e=self._stats['eccentricity'])
        self._stats['radius'] = nx.algorithms.radius(graph, e=self._stats['eccentricity'])

        try:
            self._stats['non_randomness'], self._stats['non_randomness_rel'] = nx.algorithms.non_randomness(graph)
        except ZeroDivisionError:
            self._stats['non_randomness'], self._stats['non_randomness_rel'] = None, None

        try:
            self._stats['wiener_index'] = nx.algorithms.wiener_index(graph)
            self._stats['closeness_vitality'] = nx.algorithms.closeness_vitality(graph, wiener_index=self._stats['wiener_index'])
        except nx.exception.NetworkXPointlessConcept:
            self._stats['wiener_index'] = None
            self._stats['closeness_vitality'] = None

        self._stats['s_metric'] = nx.algorithms.s_metric(graph, normalized=False)

        try:
            self._stats['small_sigma'] = nx.algorithms.sigma(graph)
        except nx.exception.NetworkXError:
            self._stats['small_sigma'] = None

        try:
            self._stats['small_omega'] = nx.algorithms.omega(graph)
        except nx.exception.NetworkXError:
            self._stats['small_omega'] = None

        self._stats['simrank'] = nx.algorithms.simrank_similarity_numpy(graph)

        try:
            self._stats['rich_club_coefficient'] = nx.algorithms.rich_club_coefficient(graph)
        except nx.exception.NetworkXError:
            self._stats['rich_club_coefficient'] = None
        except IndexError:
            self._stats['rich_club_coefficient'] = None

        try:
            self._stats['reciprocity'] = nx.algorithms.reciprocity(graph)
        except nx.exception.NetworkXError:
            self._stats['reciprocity'] = None

    def _network_prop(self, prop):
        try:
            return self._stats[prop]
        except KeyError:
            self._network_stats()
            return self._stats[prop]

    @property
    def density(self):
        return self._network_prop('density')

    @property
    def degree_hist(self):
        return self._network_prop('degree_hist')

    @property
    def in_degree_hist(self):
        return self._network_prop('in_degree_hist')

    @property
    def out_degree_hist(self):
        return self._network_prop('out_degree_hist')

    @property
    def max_clique(self):
        return self._network_prop('max_clique')

    @property
    def avg_cluster(self):
        return self._network_prop('avg_cluster')

    @property
    def tree_width(self):
        return self._network_prop('tree_width')

    @property
    def assortativity(self):
        return self._network_prop('assortativity')

    @property
    def centrality_eigen(self):
        return self._network_prop('centrality_eigen')

    @property
    def centrality_katz(self):
        return self._network_prop('centrality_katz')

    @property
    def centrality_closeness(self):
        return self._network_prop('centrality_closeness')

    @property
    def centrality_load(self):
        return self._network_prop('centrality_load')

    @property
    def centrality_degree(self):
        return self._network_prop('centrality_degree')

    @property
    def centrality_info(self):
        return self._network_prop('centrality_info')

    @property
    def centrality_betweeness(self):
        return self._network_prop('centrality_betweeness')

    @property
    def centrality_current_flow_betweeness(self):
        return self._network_prop('centrality_current_flow_betweeness')

    @property
    def estrada_index(self):
        return self._network_prop('estrada_index')

    @property
    def centrality_harmonic(self):
        return self._network_prop('centrality_harmonic')

    @property
    def centrality_second_order(self):
        return self._network_prop('centrality_second_order')

    @property
    def longest_path(self):
        return self._network_prop('longest_path')

    @property
    def diameter(self):
        return self._network_prop('diameter')

    @property
    def radius(self):
        return self._network_prop('radius')

    @property
    def eccentricity(self):
        return self._network_prop('eccentricity')

    @property
    def non_randomness(self):
        return self._network_prop('non_randomness')

    @property
    def non_randomness_rel(self):
        return self._network_prop('non_randomness_rel')

    @property
    def wiener_index(self):
        return self._network_prop('wiener_index')

    @property
    def closeness_vitality(self):
        return self._network_prop('closeness_vitality')

    @property
    def s_metric(self):
        return self._network_prop('s_metric')

    @property
    def small_sigma(self):
        return self._network_prop('small_sigma')

    @property
    def small_omega(self):
        return self._network_prop('small_omega')

    @property
    def simrank(self):
        return self._network_prop('simrank')

    @property
    def rich_club_coefficient(self):
        return self._network_prop('rich_club_coefficient')

    @property
    def reciprocity(self):
        return self._network_prop('reciprocity')

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
