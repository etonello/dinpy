import sys
import os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..'))
from pprint import pprint
from pysmt.shortcuts import And, Not

from dinpy.din import attractors
from dinpy.input_din import polys
from dinpy.interaction_graphs import local_circuits
from dinpy.find_din import boolean_map, solve, circuits, attractive_cycle, is_circuit
from dinpy.find_din import multilevel, stepwise, fixed_point
from dinpy.multi_to_boolean import boolean_to_multi


def no_local_neg_circuits(f, n):
    return [Not(is_circuit(f, x, c, sign=-1)) for x in f for c in circuits(n)]

# find a Boolean network with an antipodal attractive cycle and no local negative circuits
n = 6
f = boolean_map(n)
nolnc = no_local_neg_circuits(f, n)
antip_cycle = [(1,)*k+(0,)*(n-k) for k in range(n)] + [(0,)*k+(1,)*(n-k) for k in range(n)] + [(0,)*n]
attr_cycle = attractive_cycle(f, antip_cycle)
print("Formula created.")
sols = solve(And(nolnc+[attr_cycle]), n, max_models=1)
try:
    g = next(sols)
    pprint(g)
    pprint(polys(g))
    lc = local_circuits(g)
    print("Signs of local circuits: {}".format(set([s for x in lc for c, s in lc[x]])))
    print("Attractors: {}".format(list(attractors(g))))
except StopIteration:
    print("No solutions.")

# find a multilevel network with no local negative circuits
ms = [3,3]
n = sum(ms)
f = boolean_map(n)
formula = And(no_local_neg_circuits(f, n) +
              [multilevel(f, ms), stepwise(f, ms), Not(f[(1,0,0,0,0,0)][0])] +
              [Not(fixed_point(f, x)) for x in f])
print("Formula created.")
sols = solve(formula, n, max_models=1)
try:
    sol = boolean_to_multi(next(sols), ms)
    pprint(sol)
except StopIteration:
    print("No solutions.")
