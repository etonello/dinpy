from functools import reduce
from itertools import product
from random import sample
from sympy import Symbol, Add, Mul, Poly

from .din import boolean_states, discrete_states, nc

# a discrete network is represented as a dict tuple(ints) -> tuple(ints)


### Truth tables

def read_truth_table(rows):
    # convert strings of the form "001 101" to a discrete network
    f = dict()
    for row in rows:
        x, fx = row.strip().split(' ')
        f[tuple([int(s) for s in x])] = tuple([int(s) for s in fx])
    return f


def read_truth_table_file(filename, header=False):
    # convert files containing strings of the form "001 101" to a discrete network
    f = dict()
    with open(filename, 'r') as fn:
        if header: fn.next()
        for row in fn:
            x, fx = row.strip().split(' ')
            f[tuple([int(s) for s in x])] = tuple([int(s) for s in fx])
    return f


def tt(f):
    for x in sorted(f.keys()):
        yield ''.join(map(str, x)) + ' ' + ''.join(map(str, f[x]))


def save_truth_table(f, filename, header=None):
    with open(filename, 'w') as fn:
        if header: fn.write(header + '\n')
        for t in tt(f):
            fn.write(t + '\n')


### polynomials and expressions

def polys_to_sd(polys, vs):
    n = len(vs)
    states = boolean_states(n)
    return dict((x, tuple(Poly(p,vs).subs({vs[i]:x[i] for i in range(n)})%2 for p in polys)) for x in states)


def poly(f, vs):
    n = len(vs)
    return reduce(Add, [f[v]*reduce(Mul, [vs[i] if v[i] else 1-vs[i] for i in range(n)]) for v in f]).factor()


def polys(f):
    n = nc(f)
    vs = [Symbol("x"+str(i+1)) for i in range(n)]
    return [poly(dict((v, f[v][i]) for v in f), vs) for i in range(n)], vs


### Generate random discrete network

def random_boolean_state(n):
    return random_state([1]*n)


def random_state(ms):
    # ms[i] is the maximum expression level of component i
    return tuple([sample(range(m+1), 1)[0] for m in ms])


def random_map(ms):
    # generate a random endomorphism on {0,1,...,m1}x...x{0,1,...,mn}
    # ms[i] is the maximum expression level of component i
    return {x: random_state(ms) for x in discrete_states(ms)}


def random_boolean_map(n):
    # generates a random endomorphism on {0, 1}^n
    return {x: random_boolean_state(n) for x in boolean_states(n)}

### Generate all discrete networks

def generate_maps(ms):
    states = list(discrete_states(ms))
    for p in product(states, repeat=len(states)):
        yield dict((states[i], p[i]) for i in range(len(states)))

def generate_boolean_maps(n):
    states = list(boolean_states(n))
    for p in product(states, repeat=len(states)):
        yield dict((states[i], p[i]) for i in range(len(states)))
