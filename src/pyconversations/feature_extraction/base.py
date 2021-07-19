from collections import defaultdict


class FeatureCache:

    """
    Basic class for caching features for re-use.
    This class should increase memory usage but decrease time requirements for feature extraction.
    """

    def __init__(self):
        self._cache = defaultdict(dict)

    def get(self, uid, key):
        """
        Queries a feature associated with a key and a unique identifier

        Parameters
        ----------
        uid : Hashable
            Some unique identifier associated with the object of interest (a post, a conversation)

        key : Hashable
            The name of the cached feature

        Returns
        -------
        Any
            The feature (if present). Returns None otherwise.
        """
        return self._cache[uid].get(key, None)

    def cache(self, uid, key, val):
        """
        Stores a key-value pair in the cache

        Parameters
        ----------
        uid : Hashable
            Some unique identifier associated with the object of interest (a post, a conversation)

        key : Hashable
            The name of the cached feature

        val : Any
            The value to cache

        Returns
        -------
        None
        """
        self._cache[uid][key] = val

    def wrap(self, uid, key, func, **kwargs):
        """
        Basic feature caching wrapper.
        Attempts to query the `key` for the `uid`, but if that fails (determined by a `None` value),
        then it will call the function in the `func` parameter with the arguments in `kwargs`.

        Parameters
        ----------
        uid : Hashable
            Some unique identifier associated with the object of interest (a post, a conversation)

        key : Hashable
            The name of the cached feature

        func : function
            The function to call with `kwargs`, if the query fails

        kwargs : dict
            Any additional arguments to be passed to `func`

        Returns
        -------
        Any
            The cached feature
        """
        x = self.get(uid, key)

        if x is None:
            x = func(**kwargs)
            self.cache(uid, key, x)

        return x

    def clear(self):
        """
        Clears all values in the cache

        Returns
        -------
        None
        """
        self._cache = defaultdict(dict)
