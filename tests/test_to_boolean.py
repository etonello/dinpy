#!/usr/bin/env python

"""Tests for dinpy."""

from collections import defaultdict
import unittest
from random import choice

from dinpy.input_din import read_truth_table, random_state, random_map, random_boolean_map, random_boolean_state
from dinpy.din import boolean_states, to_stepwise, is_trap_domain, attractors, diff_inds, neigh
from dinpy.interaction_graphs import local_int_graph, local_circuits, global_circuits
from dinpy.multi_to_boolean import multi_to_boolean_adm, multi_to_boolean, to_boolean, boolean_to_sum, admissible_sum_vect, binarisation
from dinpy.multi_to_boolean import multi_level_to_bool, bool_vars_to_multi, admissible_states, to_multi, to_boolean_vect, boolean_to_multi


class TestToBool(unittest.TestCase):
    def test_to_boolean(self):
        self.assertEqual(to_boolean(0, 0), ())
        self.assertEqual(to_boolean(0, 3), (0,0,0))
        self.assertEqual(to_boolean(1, 1), (1,))
        self.assertEqual(to_boolean(1, 2), (1,0))
        self.assertEqual(to_boolean(1, 5), (1,0,0,0,0))
        self.assertEqual(to_boolean(2, 2), (1,1))
        self.assertEqual(to_boolean(2, 3), (1,1,0))
        with self.assertRaises(ValueError):
            to_boolean(1,0)
        with self.assertRaises(ValueError):
            to_boolean(7,2)

    def test_admissible(self):
        states = [(0,0,0,0,0), (0,0,0,1,0), (0,0,0,1,1),
                  (1,0,0,0,0), (1,0,0,1,0), (1,0,0,1,1),
                  (1,1,0,0,0), (1,1,0,1,0), (1,1,0,1,1),
                  (1,1,1,0,0), (1,1,1,1,0), (1,1,1,1,1)]
        self.assertEqual(set(admissible_states([3,2])), set(states))

    def test_indices(self):
        ms = [3,2]
        mltb, btml = multi_level_to_bool(ms), bool_vars_to_multi(ms)
        self.assertEqual(len(btml.keys()), sum(ms))
        self.assertEqual(mltb[(1,1)], 1)
        self.assertEqual(btml[1], (1,1))
        self.assertEqual(btml[3], (1,3))
        self.assertEqual(mltb[(2,2)], 5)

    def test_to_multi(self):
        ms = [3,2,4]
        self.assertEqual(to_multi((1,0,1,1,1,1,0,1,1), ms), (2,2,3))
        self.assertEqual(to_multi((0,0,0,0,1,0,0,1,1), ms), (0,1,2))
        for i in range(5):
            x = random_state(ms)
            self.assertEqual(x, to_multi(to_boolean_vect(x, ms), ms))

    def test_multi_to_boolean_adm(self):
        f = read_truth_table(["0 1", "1 2", "2 0"])
        fb = multi_to_boolean_adm(f)
        d = {(0,0): (1,0), (1,0): (1,1), (1,1): (0,0)}
        self.assertEqual(fb, d)
        self.assertEqual(f, boolean_to_multi(fb, [2]))
        ms = [3,1,3]
        f = random_map(ms)
        self.assertEqual(f, boolean_to_multi(multi_to_boolean(f), ms))
        with self.assertRaises(ValueError):
            boolean_to_multi({(0,0): (1,0), (0,1): (1,0), (1,0): (0,1), (1,1): (0,0)}, [2])

    def test_stepwise_adm(self):
        ms = [3,3,2]
        n = sum(ms)
        f = to_stepwise(random_map(ms))
        fb = multi_to_boolean_adm(f)
        g = random_boolean_map(sum(ms))
        adms = admissible_states(ms)
        na = len(adms)
        Fb = {x: fb[x] if x in fb else random_boolean_state(n) for x in boolean_states(n)}
        self.assertTrue(is_trap_domain(Fb, adms))
        Fb = {x: fb[x] if x in fb else adms[choice(range(na))] for x in boolean_states(n)}
        self.assertTrue([all(x in adms for x in a) for a in attractors(Fb)])

    def test_admissible_sum(self):
        self.assertEqual(admissible_sum_vect((1,1,0,0,1), (3,2)), (1,1,0,1,0))
        self.assertEqual(admissible_sum_vect((1,0), (2,)), (1,0))
        self.assertEqual(admissible_sum_vect((1,0,1,0,1,0,1,0), (4,3,1)), (1,1,0,0,1,1,0,0))
        with self.assertRaises(ValueError):
            admissible_sum_vect((1,1,0,0,1), (3,4))

    def test_boolean_to_sum(self):
        self.assertEqual(boolean_to_sum((1,1,0,0,1), (3,2)), (2,1))
        self.assertEqual(boolean_to_sum((1,0), (2,)), (1,))
        with self.assertRaises(ValueError):
            boolean_to_sum((1,1,0,0,1), (3,4))

    def test_binarise(self):
        f = read_truth_table(["0 2", "1 1", "2 0"])
        Bf = read_truth_table(["00 11", "10 10", "01 01", "11 00"])
        self.assertEqual(binarisation(f), Bf)
        f = read_truth_table(["0 0", "1 1", "2 0"])
        Bf = {x: (0,0) if x==(1,1) else x for x in boolean_states(2)}
        self.assertEqual(binarisation(f), Bf)

    def test_multi_to_boolean(self):
        f = read_truth_table(["0 2", "1 1", "2 0"])
        Fb = read_truth_table(["00 11", "10 10", "01 10", "11 00"])
        uFb = read_truth_table(["00 10", "10 10", "01 10", "11 10"])
        self.assertEqual(multi_to_boolean(f), Fb)
        self.assertEqual(multi_to_boolean(f, unit=True), uFb)
        f = read_truth_table(["0 0", "1 1", "2 0"])
        Fb = read_truth_table(["00 00", "10 10", "01 10", "11 00"])
        uFb = read_truth_table(["00 00", "10 10", "01 10", "11 10"])
        self.assertEqual(multi_to_boolean(f), Fb)
        self.assertEqual(multi_to_boolean(f, unit=True), uFb)

    def test_lig_adm(self):
        # interaction graph of f and fb
        ms = [3,2,4]
        n = len(ms)
        f = random_map(ms)
        fb = multi_to_boolean_adm(f)
        lig, ligb = local_int_graph(f, direction=True), local_int_graph(fb, direction=True)
        mltb, btml = multi_level_to_bool(ms), bool_vars_to_multi(ms)
        for x in fb:
            y = to_multi(x, ms)
            self.assertTrue((btml[j][0], btml[i][0], s1, s) in lig[y] for j, i, s1, s in ligb[x])
        for y in f:
            x = to_boolean_vect(y, ms)
            for j, i, s1, s in lig[y]:
                j1 = (j, y[j-1] + int((s1+1)/2))
                ys1 = tuple(y[h] if h+1!=j else y[h]+s1 for h in range(n))
                a, b = sorted([f[y][i-1], f[ys1][i-1]])
                for k in range(a+1,b+1):
                    self.assertTrue((mltb[j1], mltb[(i,k)], s1, s) in ligb[x])

    def test_lig_boolean(self):
        # interaction graph of fb and Fb
        ms = [3,1,2]
        n = len(ms)
        f = random_map(ms)
        fb = multi_to_boolean_adm(f)
        Fb = multi_to_boolean(f)
        ligb, ligB = local_int_graph(fb), local_int_graph(Fb)
        for x in Fb:
            psix = admissible_sum_vect(x, ms)
            for jb, ib, s in ligB[x]:
                jkp = diff_inds(admissible_sum_vect(neigh(x, jb-1), ms), psix)[0]+1
                self.assertTrue((jkp, ib, s) in ligb[psix])
        mltb, btml = multi_level_to_bool(ms), bool_vars_to_multi(ms)
        for x in fb:
            for jb, ib, s in ligb[x]:
                j = btml[jb][0]
                for t in range(ms[j-1]):
                    jt = mltb[(j,t+1)]
                    ys = [y for y in Fb if admissible_sum_vect(y, ms)==x and y[jt-1]==x[jb-1]]
                    for y in ys:
                        self.assertTrue((jt, ib, s) in ligB[y])

    def test_to_boolean_circuits(self):
        # circuits in f and Fb
        ms = [2,3,2]
        n = len(ms)
        f = random_map(ms)
        Fb = multi_to_boolean(f)
        # local
        lc, lcb = local_circuits(f), local_circuits(Fb)
        lcb_signs = defaultdict(list)
        for y in lcb:
            x = admissible_sum_vect(y, ms)
            lcb_signs[x] = lcb_signs[x] + [s for c, s in lcb[y]]
        for x in lc:
            for _, s in lc[x]:
                self.assertTrue(s in lcb_signs[to_boolean_vect(x, ms)])
        for y in lcb:
            if len(lcb[y])>0:
                mx = to_multi(y, ms)
                self.assertTrue(len(lc[mx])>0)
                if -1 in lcb[y]:
                    self.assertTrue(-1 in [s for _, s in lc[mx]])
        # global
        sgc, sgcb = [s for _, s in global_circuits(f)], [s for _, s in global_circuits(Fb)]
        if 1 in sgc:
            self.assertTrue(1 in sgcb)
        if -1 in sgc:
            self.assertTrue(-1 in sgcb)
        if len(sgcb)>0:
            self.assertTrue(len(sgc)>0)
        # counterexample positive sign
        f = read_truth_table(["000 100", "001 100", "010 100", "011 100",
                              "020 010", "021 110", "100 111", "101 111",
                              "110 110", "111 120", "120 010", "121 120"])
        Fb = multi_to_boolean(f)
        lc, lcb = local_circuits(f), local_circuits(Fb)
        x = (1,1,0)
        y = to_boolean_vect(x, [1,2,1])
        slc, slcb = [s for _, s in lc[x]], [s for _, s in lcb[y]]
        self.assertTrue((1 in slcb) and (1 not in slc))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestToBool)
    unittest.TextTestRunner(verbosity=2).run(suite)
