from itertools import combinations, permutations
from pysmt.shortcuts import And, EqualsOrIff, Iff, Not, Or, Symbol, Solver, Implies, Bool

from .din import boolean_states, nc, diff_inds, neigh
from .multi_to_boolean import multi_level_to_bool, admissible_states, admissible_sum_vect

### use pysmt to write Boolean formulas expressing properties of Boolean networks
### use a SAT solver to find one or all Boolean networks satisfying the requirements

# f[x] = y
def map_state(f, x, y):
    return And([f[x][i] if y[i] else Not(f[x][i]) for i in range(len(x))])

# f[x] = x
def fixed_point(f, x):
    return map_state(f, x, x)

# f[x] in xs
def map_state_set(f, x, xs):
    return Or([map_state(f, x, y) for y in xs])

# y successor for x in asynchronous dynamics
def succ(f, x, y):
    I = diff_inds(x, y)
    if len(I)!=1:
        raise ValueError("{} and {} are not neighbours.".format(x, y))
    return f[x][I[0]] if y[I[0]] else Not(f[x][I[0]])

# xs possible successors for x in asynchronous dynamics
def succ_set(f, x, xs):
    return Or([succ(f, x, y) for y in xs])

# orbit
def orbit(f, path):
    return And([succ(f, path[i], path[i+1]) for i in range(len(path)-1)])

# trap set
def trap_set(f, xs):
    constraints = []
    for x in xs:
        for i in range(len(x)):
            if neigh(x, i) not in xs:
                constraints.append(f[x][i] if x[i] else Not(f[x][i]))
    return And(constraints)

# attractive cycle
def attractive_cycle(f, c):
    if c[0]!=c[-1]:
        raise ValueError("{} must be a cycle.".format(c))
    return And(orbit(f, c), trap_set(f, c))

# interaction edges
def x0(x, j):
    return tuple(x[i] if i!=j else 0 for i in range(len(x)))

def x1(x, j):
    return tuple(x[i] if i!=j else 1 for i in range(len(x)))

def label(f, x, i, j):
    return (f[x0(x, j)][i], f[x1(x, j)][i])

def is_neg(l):
    return And(l[0], Not(l[1]))

def is_zero(l):
    return Iff(l[0], l[1])

def is_pos(l):
    return And(Not(l[0]), l[1])

def edge(f, x, j, i, s):
    # edge at x from j to i of sign s
    i, j = i-1, j-1
    if not s in [-1,0,+1]:
        raise ValueError("Invalid sign. Sign must be -1,0 or 1.")
    if s==-1:
        return is_neg(label(f, x, i, j))
    if s==0:
        return is_zero(label(f, x, i, j))
    if s==+1:
        return is_pos(label(f, x, i, j))

# local
def local_edges(f, edges, only=True):
    # local edges is a dictionary
    # state -> edges (j,i,s) in the interaction graph
    # if only=False, do not exclude other edges
    if only:
        n = len(list(f.keys())[0])
        exclude = [Not(edge(f, x, j, i, s))
                   for i in range(1,n+1) for j in range(1,n+1) for s in [-1,+1] for x in edges
                   if (j,i,s) not in edges[x]]
    return And([edge(f, x, j, i, s) for x in edges for j, i, s in edges[x]] + \
               (exclude if only else []))

# global
def global_edges(f, edges, only=True, all_states=True):
    if all_states:
        ledges = {x: edges for x in f}
        return local_edges(f, ledges, only)
    else:
        if only:
            n = len(list(f.keys())[0])
            exclude = [Not(edge(f, x, j, i, s))
                       for i in range(1,n+1) for j in range(1,n+1) for s in [-1,+1] for x in f
                       if (j,i,s) not in edges]
        return And([Or([edge(f, x, j, i, s) for x in f]) for j, i, s in edges] +
                   (exclude if only else []))

# circuits

def circuits(n):
    # circuits in a complete graph with n nodes
    return [[c[0]] + list(p) for h in range(1, n+1)
                             for c in combinations(range(1, n+1), h)
                             for p in permutations(c[1:])]

def is_circuit(f, x, c, sign=None):
    edges = list(zip(c, c[1:]+[c[0]]))
    k = len(c)+1
    if sign==-1:
        # an odd number of edges are negative
        return Or([And([edge(f, x, j, i, -1) for j,i in ls] +
                       [edge(f, x, j, i, +1) for j,i in edges if (j,i) not in ls])
                   for m in range(1,k+1,2) for ls in combinations(edges, m)])
    if sign==+1:
        # an even number of edges are negative
        return Or([And([edge(f, x, j, i, -1) for j,i in ls] +
                       [edge(f, x, j, i, +1) for j,i in edges if (j,i) not in ls])
                   for m in range(0,k+1,2) for ls in combinations(edges, m)])
    # all edges exist
    return And([Not(edge(f, x, j, i, 0)) for j,i in edges])

def is_global_circuit(f, c, sign=None):
    edges = list(zip(c, c[1:]+[c[0]]))
    k = len(c)+1
    if sign==-1:
        # an odd number of edges are negative
        return Or([And([Or(edge(f, x, j, i, -1) for x in f) for j,i in ls] +
                       [Or(edge(f, x, j, i, +1) for x in f) for j,i in edges if (j,i) not in ls])
                   for m in range(1,k+1,2) for ls in combinations(edges, m)])
    if sign==+1:
        # an even number of edges are negative
        return Or([And([Or(edge(f, x, j, i, -1) for x in f) for j,i in ls] +
                       [Or(edge(f, x, j, i, +1) for x in f) for j,i in edges if (j,i) not in ls])
                   for m in range(0,k+1,2) for ls in combinations(edges, m)])
    # all edges exist
    return And([Not(And(edge(f, x, j, i, 0) for x in f)) for j,i in edges])

def path_indices(path):
    inds = []
    for i in range(len(path)-1):
        I = diff_inds(path[i], path[i+1])
        if len(I)!=1:
            raise ValueError("{} and {} are not neighbours.".format(path[i], path[i+1]))
        inds.append(I[0]+1)
    return inds

def is_path_circuit(f, path, c, sign=None):
    edges = list(zip(c, c[1:]+[c[0]]))
    k = len(c)+1
    inds = path_indices(path)
    if any(i not in inds for i in c):
        return Bool(False)
    if sign==-1:
        # an odd number of edges are negative
        return Or([And([edge(f, path[inds.index(j)], j, i, -1) for j,i in ls] +
                       [edge(f, path[inds.index(j)], j, i, +1) for j,i in edges if (j,i) not in ls])
                   for m in range(1,k+1,2) for ls in combinations(edges, m)])
    if sign==+1:
        # an even number of edges are negative
        return Or([And([edge(f, path[inds.index(j)], j, i, -1) for j,i in ls] +
                       [edge(f, path[inds.index(j)], j, i, +1) for j,i in edges if (j,i) not in ls])
                   for m in range(0,k+1,2) for ls in combinations(edges, m)])
    # all edges exist
    return And([Not(edge(f, path[inds.index(j)], j, i, 0)) for j,i in edges])

# multilevel networks

def multilevel(f, ms):
    mltb = multi_level_to_bool(ms)
    # impose all states mapped to admissible
    a = [Implies(f[x][mltb[(i,j+1)]-1], f[x][mltb[(i,j)]-1])
         for x in f for i in range(1,len(ms)+1) for j in range(1,ms[i-1])]
    # impose f(x) = f(adm(x))
    b = [Iff(f[admissible_sum_vect(x, ms)][i], f[x][i]) for i in range(sum(ms)) for x in f]
    return And(a+b)

def stepwise(f, ms):
    adms = admissible_states(ms)
    mltb = multi_level_to_bool(ms)
    # if xij=0, fi(j+1) must be 0
    c0 = [Not(f[x][mltb[(i,j+1)]-1]) for i in range(1,len(ms)+1) for j in range(1,ms[i-1]) for x in adms if x[mltb[(i,j)]-1]==0]
    # if xij=1, fi(j-1) must be 1
    c1 = [f[x][mltb[(i,j-1)]-1] for i in range(1,len(ms)+1) for j in range(2,ms[i-1]+1) for x in adms if x[mltb[(i,j)]-1]==1]
    return And(c0+c1)

# solver and auxiliary functions

def variables(n):
    # the Boolean network is represented by n*2^n Boolean variables
    return [[Symbol("x{}_{}".format(i+1,j+1)) for j in range(n)] for i in range(2**n)]


def boolean_map(n):
    return dict(zip(boolean_states(n), variables(n)))


def to_bn(model, n):
    states, vs = boolean_states(n), variables(n)
    return {x: tuple(1 if model[v].is_true() else 0 for v in vsi) for x, vsi in zip(states, vs)}


def solve(formula, n, max_models=None, solver="msat"):
    s = Solver(name=solver)
    st = s.is_sat(formula)
    if st:
        vs = [x for xs in variables(n) for x in xs]
        k = 0
        s.add_assertion(formula)
        while s.solve() and ((not max_models) or k<max_models):
            k = k+1
            model = s.get_model()
            s.add_assertion(Not(And([EqualsOrIff(v, model[v]) for v in vs])))
            yield to_bn(model, n)
