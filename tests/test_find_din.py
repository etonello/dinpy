#!/usr/bin/env python

"""Tests for dinpy."""

import unittest
from pysmt.shortcuts import And, Not

from dinpy.din import is_trap_domain, attractive_cycles, is_stepwise
from dinpy.interaction_graphs import local_int_graph_state, local_int_graph, global_int_graph, local_circuits, path_circuits, global_circuits, path_graph
from dinpy.multi_to_boolean import boolean_to_multi, multi_level_to_bool
from dinpy.find_din import boolean_map, map_state, map_state_set, fixed_point, solve, succ, succ_set, orbit, trap_set, attractive_cycle
from dinpy.find_din import edge, local_edges, global_edges, circuits, is_circuit, multilevel, stepwise, is_path_circuit, path_indices, is_global_circuit


class TestFindDin(unittest.TestCase):
    def test_map_state(self):
        f = boolean_map(1)
        formula = And(fixed_point(f, (0,)), map_state(f, (0,), (1,)))
        self.assertEqual(len(list(solve(formula, 1))), 0)
        self.assertEqual(len(list(solve(fixed_point(f, (0,)), 1))), 2)
        self.assertEqual(len(list(solve(map_state_set(f, (0,), [(0,), (1,)]), 1))), 4)
        self.assertEqual(len(list(solve(map_state_set(f, (0,), [(0,), (1,)]), 1, max_models=3))), 3)

    def test_succ(self):
        n = 3
        f = boolean_map(n)
        formula = succ(f, (0,0,0), (0,1,0))
        sols = solve(formula, n, max_models=4)
        for sol in sols:
            self.assertEqual(sol[(0,0,0)][1], 1)
        formula = succ_set(f, (1,1,0), [(0,1,0), (1,1,1)])
        sols = solve(formula, n, max_models=4)
        for sol in sols:
            self.assertTrue(sol[(1,1,0)][0]==0 or sol[(1,1,0)][2]==1)

    def test_orbit(self):
        n = 3
        f = boolean_map(n)
        formula = orbit(f, [(0,0,0), (0,1,0), (1,1,0), (1,0,0)])
        sol = next(solve(formula, n, max_models=1))
        self.assertEqual(sol[(0,0,0)][1], 1)
        self.assertEqual(sol[(0,1,0)][0], 1)
        self.assertEqual(sol[(1,1,0)][1], 0)
        with self.assertRaises(ValueError):
            orbit(f, [(1,1,0), (1,0,0), (1,1,1), (0,1,1)])
        n = 4
        f = boolean_map(n)
        c = [(0,1,0,0), (0,1,1,0), (0,1,1,1), (1,1,1,1), (1,1,0,1), (1,1,0,0), (0,1,0,0)]
        formula = orbit(f, c)
        sol = solve(formula, n, max_models=1)
        self.assertEqual(next(sol)[(0,1,0,0)][2], 1)

    def test_trap_set(self):
        n = 3
        f = boolean_map(n)
        xs = [(0,0,0), (0,1,0), (1,1,1)]
        formula = trap_set(f, xs)
        sol = next(solve(formula, n, max_models=1))
        self.assertTrue(is_trap_domain(sol, xs))

    def test_attractive_cycle(self):
        n = 4
        f = boolean_map(n)
        with self.assertRaises(ValueError):
            attractive_cycle(f, [(1,1,0), (1,0,0), (1,1,1), (0,1,1)])
        c = [(0,1,0,0), (0,1,1,0), (0,1,1,1), (1,1,1,1), (1,1,0,1), (1,1,0,0), (0,1,0,0)]
        formula = attractive_cycle(f, c)
        sols = solve(formula, n, max_models=10)
        for sol in sols:
            self.assertEqual(sol[(0,1,0,0)][2], 1)
            self.assertTrue(any(sorted(a)==sorted(c[:-1]) for a in attractive_cycles(sol)))

    def test_edge(self):
        n, ms = 3, [1]*3
        f = boolean_map(n)
        formula = And(edge(f, (0,0,1), 2, 3, -1),
                      edge(f, (0,0,0), 3, 1, 0),
                      edge(f, (0,1,1), 1, 1, +1))
        sols = solve(formula, n, max_models=10)
        for sol in sols:
            self.assertTrue(sol[(0,0,1)][2]==1 and sol[(0,1,1)][2]==0)
            self.assertTrue(sol[(0,0,0)][0]==sol[(0,0,1)][0])
            self.assertTrue(sol[(0,1,1)][0]==0 and sol[(1,1,1)][0]==1)
            self.assertTrue((2,3,-1) in local_int_graph_state(sol, (0,0,1), ms))
            self.assertTrue((3,1,-1) not in local_int_graph_state(sol, (0,0,0), ms))
            self.assertTrue((3,1,+1) not in local_int_graph_state(sol, (0,0,0), ms))
            self.assertTrue((1,1,+1) in local_int_graph_state(sol, (0,1,1), ms))
        with self.assertRaises(ValueError):
            solve(edge(f, (0,1,1), 2, 3, 2), n)

    def test_local_graph(self):
        n = 1
        f = boolean_map(n)
        formula = local_edges(f, {(0,): [(1,1,1)]})
        sols = solve(formula, n)
        self.assertEqual(len(list(sols)), 1)
        n = 2
        f = boolean_map(n)
        formula = local_edges(f, {(0,0): [(1,2,1), (2,1,-1)]})
        sols = solve(formula, n, max_models=10)
        for sol in sols:
            lig = local_int_graph(sol)
            self.assertTrue((1,2,1) in lig[(0,0)])
            self.assertTrue((2,1,-1) in lig[(0,0)])
            self.assertFalse((1,2,-1) in lig[(0,0)])
            self.assertFalse((2,1,+1) in lig[(0,0)])
        f = boolean_map(n)
        formula = local_edges(f, {(0,0): [(1,2,1), (2,1,-1)], (0,1): [(2,1,1)]}, only=False)
        sols = solve(formula, n, max_models=10)
        with self.assertRaises(StopIteration):
            next(sols)
        formula = local_edges(f, {(0,0): [(1,2,1), (2,1,-1)], (1,1): [(2,1,1)]}, only=False)
        sols = solve(formula, n, max_models=10)
        for sol in sols:
            lig = local_int_graph(sol)
            self.assertTrue((1,2,1) in lig[(0,0)])
            self.assertTrue((2,1,-1) in lig[(0,0)])

    def test_global_graph(self):
        n = 3
        f = boolean_map(n)
        formula = global_edges(f, [(1,2,1), (2,3,1), (3,1,-1)])
        sols = solve(formula, n)
        k = 0
        for sol in sols:
            k = k + 1
            lig = local_int_graph(sol)
            for x in f:
                self.assertEqual(lig[x], [(1,2,1), (2,3,1), (3,1,-1)])
            self.assertEqual(global_int_graph(sol), [(1,2,1), (2,3,1), (3,1,-1)])
        self.assertEqual(k, 1)
        formula = global_edges(f, [(1,2,1), (2,3,1)], all_states=False)
        sols = solve(formula, n, max_models=3)
        for sol in sols:
            grg = global_int_graph(sol)
            self.assertTrue((1,2,1) in grg)
            self.assertTrue((2,3,1) in grg)
            self.assertFalse((3,1,-1) in grg)
            self.assertFalse((3,1,1) in grg)
            self.assertFalse((3,2,-1) in grg)
            self.assertFalse((3,2,1) in grg)
        formula = global_edges(f, [(1,2,1), (2,3,1), (3,1,-1)], only=False, all_states=False)
        sols = solve(formula, n, max_models=3)
        for sol in sols:
            grg = global_int_graph(sol)
            self.assertTrue((1,2,1) in grg)
            self.assertTrue((2,3,1) in grg)
            self.assertTrue((3,1,-1) in grg)

    def test_circuits(self):
        self.assertEqual(circuits(1), [[1]])
        self.assertEqual(circuits(2), [[1],[2],[1,2]])
        self.assertEqual(circuits(3), [[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3],[1,3,2]])

    def test_circuit_signs(self):
        n = 4
        f = boolean_map(n)
        formula = is_circuit(f, (0,1,0,0), [1,4,2])
        sol = next(solve(formula, n, max_models=1))
        self.assertTrue((([1,4,2],-1) in local_circuits(sol)[(0,1,0,0)]) or \
                        (([1,4,2],1) in local_circuits(sol)[(0,1,0,0)]))
        formula = And([is_circuit(f, x, [2,3,4], sign=-1) for x in f])
        sol = next(solve(formula, n, max_models=1))
        lcs = local_circuits(sol)
        for x in f:
            self.assertTrue(([2,3,4],-1) in lcs[x])
        formula = And([is_circuit(f, x, [1,2,3,4], sign=+1) for x in f])
        sol = next(solve(formula, n, max_models=1))
        lcs = local_circuits(sol)
        for x in f:
            self.assertTrue(([1,2,3,4],1) in lcs[x])

    def test_global_circuit(self):
        n = 4
        f = boolean_map(n)
        formula = And([Not(is_circuit(f, x, [1,2,3,4])) for x in f] + [is_global_circuit(f, [1,2,3,4])])
        sol = next(solve(formula, n, max_models=1))
        lcs = local_circuits(sol)
        gcs = global_circuits(sol)
        self.assertTrue(([1,2,3,4],1) in gcs or ([1,2,3,4],-1) in gcs)
        for x in f:
            self.assertFalse(([1,2,3,4],1) in lcs[x])
            self.assertFalse(([1,2,3,4],-1) in lcs[x])
        formula = And([Not(is_circuit(f, x, [2,3,4], -1)) for x in f] + [is_global_circuit(f, [2,3,4], sign=-1)])
        sol = next(solve(formula, n, max_models=1))
        lcs = local_circuits(sol)
        gcs = global_circuits(sol)
        self.assertTrue(([2,3,4],-1) in gcs)
        for x in f:
            self.assertFalse(([2,3,4],-1) in lcs[x])
        formula = And([Not(is_circuit(f, x, [1,4,3], +1)) for x in f] + [is_global_circuit(f, [1,4,3], sign=+1)])
        sol = next(solve(formula, n, max_models=1))
        lcs = local_circuits(sol)
        gcs = global_circuits(sol)
        self.assertTrue(([1,4,3],+1) in gcs)
        for x in f:
            self.assertFalse(([1,4,3],+1) in lcs[x])

    def test_multilevel(self):
        ms = [2,1,4]
        n = sum(ms)
        f = boolean_map(n)
        rg = [(1,1,1), (2,1,-1)]
        rgl = [((1,1),(1,2),1), ((2,1),(1,1),-1)]
        mltb = multi_level_to_bool(ms)
        formula = And(multilevel(f, ms), global_edges(f, [(mltb[j],mltb[i],s) for j,i,s in rgl], all_states=False, only=False))
        sol = boolean_to_multi(next(solve(formula, n, max_models=1)), ms)
        self.assertTrue(all(e in global_int_graph(sol) for e in rg))

        formula = And(formula, stepwise(f, ms))
        sol = boolean_to_multi(next(solve(formula, n, max_models=1)), ms)
        self.assertTrue(all(e in global_int_graph(sol) for e in rg))
        self.assertTrue(is_stepwise(sol))

    def test_path_indices(self):
        self.assertEqual(path_indices([(1,0,1),(1,0,0),(1,1,0),(0,1,0)]), [3,2,1])
        with self.assertRaises(ValueError):
            path_indices([(1,0,1),(1,1,0),(1,1,0),(0,1,0)])

    def test_circuit_path(self):
        n = 3
        f = boolean_map(n)
        path = [(0,0,0),(0,1,0),(0,1,1),(1,1,1)]
        path1 = [(0,0,0),(1,0,0)]
        formula = is_path_circuit(f, path, [1,2,3], +1)
        formula = And(formula, And([Not(is_path_circuit(f, path, c)) for c in circuits(n) if len(c)<3]))
        formula = And(formula, Not(f[(1,1,1)][2]))
        formula = And(formula, is_path_circuit(f, path1, [1], -1))
        formula = And(formula, Not(is_path_circuit(f, path1, [3])))
        sols = solve(formula, n, max_models=2)
        for sol in sols:
            cs = path_circuits(sol, path)
            self.assertTrue(all(len(c[0])>=3 for c in cs))
            self.assertTrue(([1,2,3], +1) in cs)
            self.assertTrue(([1], -1) in path_circuits(sol, path1))
            self.assertTrue(([1], -1) in path_circuits(sol, path1, sign=-1))
            self.assertEqual(path_graph(sol, []), [])
            self.assertEqual(path_graph(sol, [(1,0),(1,1)]), [])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDin)
    unittest.TextTestRunner(verbosity=2).run(suite)
