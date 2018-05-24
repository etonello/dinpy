# -*- coding: utf-8 -*-

from itertools import chain

from .din import max_levels, nc, to_asymptotic, to_stepwise, boolean_states, discrete_states

### conversions

# admissible only

def to_boolean(x, m):
    if x > m:
        raise ValueError("Value out of variable limit: bound", m, ", value", x)
    return tuple(1 for i in range(x)) + tuple(0 for i in range(x, m))


def to_boolean_vect(xs, ms):
    return tuple(chain.from_iterable(map(lambda v: to_boolean(v[0], v[1]), zip(xs, ms))))


def to_multi(x, ms):
    # inverse of to_boolean_vect
    return tuple(sum(x[sum(ms[:i]):sum(ms[:i+1])]) for i in range(len(ms)))


def admissible_states(ms):
    return [to_boolean_vect(xs, ms) for xs in discrete_states(ms)]


def multi_level_to_bool(ms):
    # return dictionary that maps
    # the i-th variable and the j-th level in 0,...,mi
    # to the corresponding Boolean variable
    pairs = [(i+1, j+1) for i in range(len(ms)) for j in range(ms[i])]
    return dict(zip(pairs, range(1, len(pairs)+1)))


def bool_vars_to_multi(ms):
    # return dictionary that maps the Boolean variable to
    # the i-th variable and the j-th level in 0,...,mi it represents
    pairs = [(i+1, j+1) for i in range(len(ms)) for j in range(ms[i])]
    return dict(zip(range(1, len(pairs)+1), pairs))


def multi_to_boolean_adm(f, ms=None):
    # convert a multilevel map to Boolean,
    # only on the admissible states.
    # Boolean variables: y_ij, i=1,...,n, j=1,...,mi,
    # for each component i, y_ij = 1 if x_i >= j, 0 otherwise.
    n = nc(f)
    if not ms: ms = max_levels(f)
    return dict((to_boolean_vect(k, ms), to_boolean_vect(v, ms)) for k, v in f.items())


def boolean_to_multi(f, ms):
    # Boolean version of map that sends admissible to admissible
    adms = admissible_states(ms)
    if any(f[x] not in adms for x in adms):
        raise ValueError("Conversion only supported for networks that map admissible states to admissible states")
    return {to_multi(x, ms): to_multi(f[x], ms) for x in adms}


# stepwise and circuit preserving conversion
# Tonello, arXiv:1703.06746.

def admissible_sum_vect(x, levels):
    # admissible state with same sum by component
    if len(x)!=sum(levels):
        raise ValueError("Levels do not match length.")
    return to_boolean_vect(boolean_to_sum(x, levels), levels)


def multi_to_boolean(f, unit=False):
    # convert a multilevel map to Boolean.
    # maps each non-admissible state to the image of the admissible
    # with the same sum for each component
    levels = max_levels(f)
    fb = multi_to_boolean_adm(to_stepwise(f) if unit else f)
    n = nc(fb)
    for x in boolean_states(n):
        if x not in fb:
            fb[x] = fb[admissible_sum_vect(x, levels)]
    return fb


# asymptotic and circuit preserving conversion
# Fauré and Kaji, J. Theor. Biol. (2018), 440(71--79).

def compare_binarize(x, f, p):
    if f < p:
        return 0
    if f == p:
        return x
    if f > p:
        return 1


def boolean_to_sum(x, levels):
    if len(x)!=sum(levels):
        raise ValueError("Levels do not match length.")
    return tuple(sum(x[sum(levels[:i]):sum(levels[:i+1])]) for i in range(len(levels)))


def binarise(x, f):
    ms = max_levels(f)
    p = boolean_to_sum(x, ms)
    return tuple(chain.from_iterable(map(lambda y: compare_binarize(*y), [(x[j], f[p][i], p[i]) for j in range(sum(ms[:i]), sum(ms[:i+1]))])
                 for i in range(len(ms))))


def binarisation(f):
    f = to_asymptotic(f)
    levels = max_levels(f)
    states = boolean_states(sum(levels))
    return dict((x, binarise(x, f)) for x in states)
