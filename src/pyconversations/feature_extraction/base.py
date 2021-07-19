from collections import defaultdict


class FeatureCache:

    def __init__(self):
        self._cache = defaultdict(dict)

    def get(self, uid, key):
        return self._cache[uid].get(key, None)

    def cache(self, uid, key, val):
        self._cache[uid][key] = val

    def clear(self):
        self._cache = defaultdict(dict)
