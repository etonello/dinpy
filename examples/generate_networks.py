import sys
import os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..'))
from pprint import pprint
from pysmt.shortcuts import And

from dinpy.din import attractors
from dinpy.interaction_graphs import global_int_graph, local_int_graph
from dinpy.find_din import boolean_map, global_edges, local_edges, solve, orbit

# find Boolean networks with a given interaction graph
n = 4
f = boolean_map(n)
edges = [(1,2,1), (2,3,-1), (3,4,1), (4,1,-1)]
print("Imposing global interaction graph: {}".format(edges))
sols = solve(global_edges(f, edges), n)
for g in sols:
    print("Interaction graph: {}".format(global_int_graph(g)))
    print("Attractors: {}".format(list(attractors(g))))

edges = [(1,3,1), (3,2,1), (3,4,1), (4,1,-1)]
print("Imposing global interaction graph: {}".format(edges))
sols = solve(global_edges(f, edges), n)
for g in sols:
    print("Interaction graph: {}".format(global_int_graph(g)))
    print("Attractors: {}".format(list(attractors(g))))

# find Boolean networks with given local graphs and orbit
edges = {(0,0,0,1): [], (1,)*n: [(1,1,-1)]}
print("Imposing local interaction graph: {}".format(edges))
path = [(0,)*n, (0,1,0,0), (1,1,0,0), (1,0,0,0)]
print("Imposing orbit in asynchronous dynamics: {}".format(path))
formula = And(local_edges(f, edges), orbit(f, path))
sol = solve(formula, n, max_models=1)
try:
    sol = next(sol)
    print("Example:")
    pprint(sol)
    print("Local interaction graph:")
    pprint(local_int_graph(sol))
except StopIteration:
    print("No solutions.")
