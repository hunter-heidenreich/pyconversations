from collections import Counter
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize


def rebound(x, bounds):
    ex = list(x)
    for i, bound in enumerate(bounds):
        if x[i] > bound[1]:
            ex[i] = 0.99 * (bound[1] - bound[0]) + bound[0]  # sum(bound)/2 # bound[1]
        elif x[i] < bound[0]:
            ex[i] = 0.01 * (bound[1] - bound[0]) + bound[0]  # sum(bound)/2 # bound[0]
    return tuple(ex)


def resonator_simple(k1, fs, rs):
    N = rs[-1]
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN

    if not ravg:
        return np.zeros_like(rs)

    sum_fs = sum(fs)
    Mavg = sum_fs * k1

    if Mavg <= 1:
        return np.zeros_like(rs)

    if not sum_fs:
        return np.zeros_like(rs)

    theta = np.log10(fs[0] * Mavg / sum_fs) / np.log10(Mavg)
    Navg = (1 - theta) * Mavg
    fhat = ((rs - theta) ** (-theta)) * (1 - (1 + ravg / rs) ** (-Navg / ravg))

    return fhat


def NLL_simple(x, fs, rs, bounds):
    x = rebound([x], bounds)
    fhat = resonator_simple(*x, fs, rs)

    M = sum(fhat)
    if not M:
        return 0

    return -(fs * np.log10(fhat / M)).sum()


def params_simple(fs, rs, x0):
    bounds = [(0.001, 1)]
    result = minimize(NLL_simple, x0=x0, method='Nelder-Mead', tol=1e-1,
                      options={'maxiter': 999}, args=(fs, rs, bounds))
    return rebound(result.x, bounds)[0]


def mixing(frequency):
    ws, fs = map(np.array, zip(*frequency.most_common()))
    rs = np.arange(1, len(ws) + 1)
    # ranks = {w: ix + 1 for ix, w in enumerate(ws)}

    sizeranks = defaultdict(list)
    for f, r in zip(fs, rs):
        sizeranks[f] = max([r, 0 if (f not in sizeranks) else sizeranks[f]])

    # regresses the avg doc size and THEN computes theta.
    N = len(fs)
    M = sum(fs)
    HN = (1 / np.arange(1, N + 1)).sum()
    ravg = N / HN
    srs = np.array([sizeranks[f] for f in fs])
    k1 = params_simple(fs, srs, N / M)
    Mavg = M * float(k1)

    if not ravg:
        return {
            'k1':      float(k1),
            'theta':   float(0),
            'entropy': float(0),
            'N_avg':   float(0),
            'M_avg':   float(Mavg),
        }

    if Mavg <= 1:
        return {
            'k1':      float(k1),
            'theta':   float(0),
            'entropy': float(0),
            'N_avg':   float(0),
            'M_avg':   float(Mavg),
        }

    if not M:
        return {
            'k1':      float(k1),
            'theta':   float(0),
            'entropy': float(0),
            'N_avg':   float(0),
            'M_avg':   float(Mavg),
        }

    theta = np.log10(fs[0] * Mavg / M) / np.log10(Mavg)
    Navg = (1 - theta) * Mavg

    def _f(rx):
        return ((rx - theta) ** (-theta)) * (1 - (1 + ravg / rx) ** (-Navg / ravg))

    fmodel = _f

    fhat = fmodel(np.array([sizeranks[f] for f in fs]))
    fnorm = sum(fhat)
    phat = fhat / fnorm
    entropy = -(fs / M).dot(np.log10(phat)) / np.log10(N)

    # return k1, theta, entropy, Navg, Mavg
    return {
        'k1':      float(k1),
        'theta':   float(theta),
        'entropy': float(entropy),
        'N_avg':   float(Navg),
        'M_avg':   float(Mavg),
    }


def novelty(frequency):
    ws, fs = map(np.array, zip(*frequency.most_common()))
    rs = np.arange(1, len(ws) + 1)

    nums = Counter(fs)
    fsum = fs.cumsum()
    mn = (fsum / fs)

    # this complex object specifies the rank-endpoints for plateaux
    srs = {s: (min(rs[fs == s]), max(rs[fs == s])) for s in nums}
    As1 = np.ones(len(fs))
    As2 = np.ones(len(fs))
    for f in sorted(nums, reverse=True)[1:]:
        f_high = fs[srs[f][0] - 2]
        f_low = fs[srs[f][1] - 1]
        f_quotient = f_high / f_low
        mn_pairs = [(mn_high, mn_low)
                    for mn_high in [np.max(mn[fs == f_high])]
                    for mn_low in [np.max(mn[fs == f_low])]]
        As1[fs == f] = 1 + np.mean(np.log10(f_quotient) /
                                   np.log10([mn_high / mn_low
                                             for mn_high, mn_low in mn_pairs]))
        As2[fs == f] = nums[f] / np.array([mn_low - mn_high
                                           for mn_high, mn_low in mn_pairs])
    return 2 / (1 / As1 + 1 / As2)
