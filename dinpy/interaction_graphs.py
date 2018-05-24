from functools import reduce
from itertools import chain, product
from networkx import DiGraph, MultiDiGraph, simple_cycles
from operator import mul

from .din import nc, max_levels, sign, diff_inds

### Interaction graphs

# local interaction graph is dict state -> list of edges
# an edge is a triplet (source, target, sign)

def local_int_graph_state(f, x, ms, I=None, direction=False):
    edges = []
    n = len(x)
    for i in range(n):
        for j in (I if I else range(n)):
            for s1 in [-1, 1]:
                if x[j]+s1<=ms[j] and x[j]+s1>=0:
                    y = tuple([x[h] if h!=j else x[h]+s1 for h in range(n)])
                    if y in f:
                        s = s1*sign(f[y][i]-f[x][i])
                        if s!=0:
                            if direction:
                                edges.append((j+1, i+1, s1, s))
                            else:
                                edges.append((j+1, i+1, s))
    return sorted(list(set(edges)))


def local_int_graph_attr_state(f, x, ms, I=None, direction=False):
    # interaction graph of Definition 5 of
    # Richard (2010) Adv. Appl. Math. 44(378-392)
    n = len(ms)
    edges = []
    for i in range(n):
        for j in (I if I else range(n)):
            Fjx = tuple(x[k] if k!=j else f[x][j] for k in range(n))
            if Fjx in f:
                b = sign(f[Fjx][i]-Fjx[i])
                if sign(f[x][i]-x[i])!=b:
                    s = sign(f[x][j]-x[j])*b
                    if s!=0:
                        edges.append((j+1, i+1, s))
    return sorted(list(set(edges)))


def local_int_graph(f, graph=local_int_graph_state, I=None, direction=False):
    n = nc(f)
    ms = max_levels(f)
    edges = dict((x, graph(f, x, ms, I, direction)) for x in f)
    return edges


def path_graph(f, path):
    if any(p not in f for p in path):
        return []
    n = nc(f)
    ms = max_levels(f)
    edges = []
    for i in range(len(path)-1):
        j = diff_inds(path[i],path[i+1])[0]
        x = path[i]
        for i in range(n):
            for s1 in [-1, 1]:
                if x[j]+s1<=ms[j] and x[j]+s1>=0:
                    y = tuple([x[h] if h!=j else x[h]+s1 for h in range(n)])
                    if y in f:
                        s = s1*sign(f[y][i]-f[x][i])
                        if s!=0:
                            edges.append((j+1, i+1, s))
    return sorted(list(set(edges)))


# global interaction graph is a list of edges

def global_int_graph(f, graph=local_int_graph_state, I=None, direction=False):
    if graph==nu_int_graph:
        lg = nu_int_graph(f)
    else:
        lg = local_int_graph(f, graph=graph, I=I, direction=direction)
    return sorted(list(set(e for x in lg for e in lg[x])))


# non-usual and other variants

def nu_int_graph_states(f, x, y, ms):
    n = len(x)
    edges = []
    for i in range(n):
        for j in range(n):
            if x[j]!=y[j]:
                epsj = sign(y[j] - x[j])
                xepsj = tuple([x[k] if k!=j else x[j] + epsj for k in range(n)])
                xiepsi = x[i] + float(sign(y[i] - x[i]))/2
                s = epsj * sign(f[xepsj][i] - f[x][i])
                mn, mx = min(f[xepsj][i], f[x][i]), max(f[xepsj][i], f[x][i])
                if s != 0 and mn < xiepsi and xiepsi < mx:
                    edges.append((j+1, i+1, s))
    return sorted(list(set(edges)))


def nu_int_graph(f):
    # non-usual interaction graph based on the definition of non-usual Jacobian matrix in
    # Richard and Comet, Discrete Appl. Math. 155(2007) 2403-2413.
    n = nc(f)
    ms = max_levels(f)
    return dict(((x, y), nu_int_graph_states(f, x, y, ms)) for x in f for y in f if x != y)


### Circuits

def circuit_label_multi(G, circuit):
    paths = [list(chain(*[k.values() for k in G.get_edge_data(i, j).values()]))
             for i, j in zip(circuit, circuit[1:] + [circuit[0]])]
    return list(set([reduce(mul, c, 1) for c in product(*paths)]))


def cycles_from_edges(edges):
    G = MultiDiGraph()
    for e in edges:
        G.add_edge(e[0], e[1], label=e[2])
    cycles = simple_cycles(G)
    return [(c, s) for c in cycles for s in circuit_label_multi(G, c)]


def local_circuits(f, graph=local_int_graph_state, I=None, direction=False, sign=None, at=None):
    if graph==nu_int_graph:
        if at: Gf = nu_int_graph_states(f, at[0], at[1], ms = max_levels(f))
        else: Gf = nu_int_graph(f)
    else:
        if at: Gf = local_int_graph_state(f, at, max_levels(f), I, direction)
        else: Gf = local_int_graph(f, graph, I, direction)
    if sign:
        return dict((x, [e for e in cycles_from_edges(Gf[x]) if e[-1]==sign]) for x in Gf)
    if at:
        return cycles_from_edges(Gf)
    return dict((x, cycles_from_edges(Gf[x])) for x in Gf)


def global_circuits(f, graph=local_int_graph_state, I=None, direction=False, sign=None):
    Gf = global_int_graph(f, graph, I, direction)
    circuits = cycles_from_edges(Gf)
    if sign:
        return sorted([e for e in circuits if e[-1]==sign])
    return sorted(circuits)


def path_circuits(f, path, sign=None):
    G = path_graph(f, path)
    circuits = cycles_from_edges(G)
    if sign:
        return sorted([e for e in circuits if e[-1]==sign])
    return sorted(circuits)
