import networkx as nx


class Conversation:

    def __init__(self, posts=None, edges=None):
        if not posts:
            posts = {}

        if not edges:
            edges = {}

        self._posts = posts  # uid -> post object
        self._edges = edges  # uid -> {reply_tos}

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

    def remove_post(self, uid):
        # remove from post dictionary
        del self._posts[uid]

        # update knowledge of edges
        del self._edges[uid]

    def __add__(self, other):
        ps = dict(self._posts)
        es = dict(self._edges)

        # merge from other
        ps.update(other.posts)
        es.update(other.edges)

        return Conversation(posts=ps, edges=es)

    def segment(self):
        """
        Analyze current post and edge configuration,
        generating sub-conversations
        if there are disparate components
        :return:
        """
        graph = nx.Graph()

        # add posts as nodes
        for uid in self._posts:
            graph.add_node(uid)

        # add reply connections as sedges
        for uid, reps in self._edges.items():
            for rid in reps:
                graph.add_edge(uid, rid)

        segments = []
        for node_set in nx.connected_components(graph):
            segments.append(
                Conversation(posts={uid: self._posts[uid] for uid in node_set},
                             edges={uid: self._edges[uid] for uid in node_set})
            )

        return segments


