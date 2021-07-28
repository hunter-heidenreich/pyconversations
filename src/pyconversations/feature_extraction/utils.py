def apply_extraction(funcs, keyset=None, ignore=None, **kwargs):
    """
    Given a set of functions and the kewords (labels) desired,
    computes the statistics

    Parameters
    ----------
    funcs : dict(str, lambda)
    keyset : None or Iterable(str)
    ignore : None or Iterable(str)
    kwargs : dict

    Returns
    -------
    dict(str, )
        The returns from the lambda functions
    """

    def _select_applicator(f):
        if 'post' in kwargs and 'convo' in kwargs:
            return f(kwargs['post'], kwargs['convo'])
        elif 'user' in kwargs and 'convo' in kwargs:
            return f(kwargs['user'], kwargs['convo'])
        elif 'post' in kwargs:
            return f(kwargs['post'])
        elif 'convo' in kwargs:
            return f(kwargs['convo'])
        else:
            raise KeyError

    if keyset is None:
        keyset = set(funcs.keys())

    if ignore is not None:
        for i in ignore:
            if i in keyset:
                keyset.remove(i)

    return {label: _select_applicator(func) for label, func in funcs.items() if label in keyset}
