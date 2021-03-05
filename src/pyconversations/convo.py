

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


