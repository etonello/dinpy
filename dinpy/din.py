from functools import reduce
from itertools import product
from networkx import attracting_components, DiGraph, simple_cycles

### some basic functions

def boolean_states(n):
    return product(*[(0, 1)]*n)


def discrete_states(ms):
    # ms[i] is the maximum expression level of component i
    return product(*[tuple(range(ms[i]+1)) for i in range(len(ms))])


def is_admissible_component(x):
    return all([x[j]<=x[i] for i in range(len(x)-1) for j in range(i+1, len(x))])


def is_admissible(x, ms):
    return all([is_admissible_component(x[sum(ms[:i]):sum(ms[:i+1])]) for i in range(len(ms))])


def nc(f):
    # number of components
    return len(list(f.keys())[0])


def max_levels(f):
    # max expression level for each component
    n = len(list(f.keys())[0])
    return [max([v[i] for v in f.keys()]) for i in range(n)]


def sign(a):
    if a==0: return 0
    return int(a/abs(a))


def neigh(x, i):
    return tuple(x[j] if j!=i else 1-x[j] for j in range(len(x)))


def diff_inds(x, y):
    return [i for i in range(len(x)) if x[i]!=y[i]]


def cube(x, I):
    # x[I] = {y | yi=xi if i not in I(x,y)}
    for y in boolean_states(len(x)):
        if all([y[i]==x[i] for i in range(len(x)) if i not in I]):
            yield y


def picube(y, x, I):
    # projection of y on the cube x[I]
    return tuple([y[i] if i in I else x[i] for i in range(len(x))])


def cubef(f, x, I):
    # restriction of f on the cube x[I]
    states = cube(x, I)
    return {y: picube(f[y], x, I) for y in states}


### stepwise, asymptotic, constant, expansive

def to_stepwise(f):
    n = nc(f)
    return dict((x, tuple(x[i] + sign(f[x][i]-x[i]) for i in range(n))) for x in f)


def asymptotic_step(x, fx, m):
    if fx > x:
        return m
    if fx == x:
        return x
    if fx < x:
        return 0


def to_asymptotic(f):
    n = nc(f)
    ms = max_levels(f)
    return dict((x, tuple([asymptotic_step(x[i], f[x][i], ms[i]) for i in range(n)])) for x in f)


def is_stepwise(f):
    return f == to_stepwise(f)


def is_asymptotic(f):
    return f == to_asymptotic(f)


def is_constant(f):
    value = list(f.values())[0]
    return all([v == value for v in f.values()])


def dist(x, y):
    return sum(abs(x[i] - y[i]) for i in range(len(x)))


def dist_set(x, xs):
    return min([dist(x, y) for y in xs])


def is_expansive(f):
    for x in f:
        for y in f:
            if dist(f[x], f[y]) > dist(x, y):
                return True
    return False


### Synchronous and asynchronous

def shift(x, eps, j):
    return tuple([x[i] if i!=j else x[i]+eps for i in range(len(x))])


def sd_to_ad(f):
    return dict((x, set([shift(x, sign(f[x][j]-x[j]), j) for j in range(len(x)) if x[j]!=f[x][j]])) for x in f)


def ad_to_sd(adf):
    return {x: tuple(1-x[j] if any(adfx[j]!=x[j] for adfx in adf[x]) else x[j] for j in range(len(x))) for x in adf}


def sd_graph(f):
    # synchronous dynamics as graph
    sdG = DiGraph()
    sdG.add_edges_from([(x, f[x]) for x in f])
    return sdG


def ad_graph(f):
    # asynchronous dynamics as graph
    adG = DiGraph()
    adf = sd_to_ad(f)
    adG.add_nodes_from(f.keys())
    adG.add_edges_from([(k, v) for k in adf.keys() for v in adf[k]])
    return adG


def update(f, i, x):
    return tuple(x[j] if j!=i-1 else f[x][i-1] for j in range(len(x)))


def update_sequence(f, word, x):
    for i in word:
        x = update(f, i, x)
    return x


def sequential(f, word):
    return {x: update_sequence(f, word, x) for x in f}


### Attractors and trap domains

def is_trap_domain(f, points):
    adf = sd_to_ad(f)
    return all([q in points for p in points for q in adf[p]])


def is_fixed(f, x, I=None):
    if not I: I = range(len(x))
    return all([f[x][i]==x[i] for i in I])


def fixed_points(f):
    return [x for x in f if x==f[x]]


def has_fixed_points(f):
    for x in f:
        if f[x]==x:
            return True
    return False


def attractors(f, synch=False):
    dG = sd_graph(f) if synch else ad_graph(f)
    return attracting_components(dG)


def cyclic_attractors(f, synch=False):
    attrs = attractors(f, synch=synch)
    for a in attrs:
        if len(a)>1:
            yield a


def attractive_cycles(f, synch=False):
    adG = ad_graph(f)
    attrs = [sorted(list(a)) for a in attractors(f, synch=synch)]
    cycles = simple_cycles(adG)
    for c in cycles:
        if sorted(c) in attrs:
            yield c


### Mirror states

def is_mirror_pair(f, x, y, defn=None):
    if x==y: return False
    I = diff_inds(x, y)
    if defn=="cube":
        g = cubef(f, x, I)
        return all([g[x][i]==1-g[y][i] for i in I])
    if defn=="boolean":
        return all([f[x][i]!=f[y][i] for i in I])
    return all([(f[x][i]<=x[i] and f[x][i]<=y[i] and f[y][i]>=x[i] and f[y][i]>=y[i]) or
                (f[y][i]<=x[i] and f[y][i]<=y[i] and f[x][i]>=x[i] and f[x][i]>=y[i]) for i in I])


def is_mirror_pair_fixed(f, x, y, defn=None):
    I = diff_inds(x, y)
    return is_mirror_pair(f, x, y, defn) and is_fixed(f, x, I) and is_fixed(f, y, I)
