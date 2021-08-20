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

    def _selects_args():
        if 'post' in kwargs and 'convo' in kwargs:
            return kwargs['post'], kwargs['convo']
        elif 'user' in kwargs and 'convo' in kwargs:
            return kwargs['user'], kwargs['convo']
        elif 'post' in kwargs:
            return kwargs['post'],
        elif 'convo' in kwargs:
            return kwargs['convo'],
        else:
            raise KeyError

    args = _selects_args()

    if keyset is not None:
        funcs = {k: funcs[k] for k in keyset if k in funcs}

    if ignore is not None:
        funcs = {k: funcs[k] for k in funcs if k not in ignore}

    return {label: func(*args) for label, func in funcs.items()}
