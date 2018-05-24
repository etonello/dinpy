#!/usr/bin/env python

"""Tests for dinpy."""

import unittest
from itertools import combinations

from dinpy.input_din import read_truth_table, read_truth_table_file, save_truth_table
from dinpy.input_din import random_state, random_boolean_state, random_map, random_boolean_map
from dinpy.input_din import polys, polys_to_sd, generate_maps, generate_boolean_maps
from dinpy.din import is_constant, is_stepwise, is_asymptotic, is_expansive, to_stepwise, to_asymptotic, boolean_states, is_admissible, discrete_states
from dinpy.din import sd_to_ad, ad_to_sd, has_fixed_points, fixed_points, is_trap_domain, attractors, attractive_cycles, cyclic_attractors
from dinpy.din import is_mirror_pair, is_mirror_pair_fixed, update, sequential, dist_set
from dinpy.interaction_graphs import local_int_graph
from dinpy.multi_to_boolean import to_boolean_vect, multi_to_boolean


class TestDin(unittest.TestCase):
    def test_empty_din(self):
        f = read_truth_table([])

    def test_read_tt_file(self):
        f = read_truth_table_file("data/test1.tt")
        self.assertTrue(is_constant(f))

    def test_save_tt_file(self):
        f = read_truth_table(["0 1", "1 2", "2 0"])
        f = save_truth_table(f, "data/test_save.tt")

    def test_poly(self):
        f = read_truth_table(["00 01", "01 01", "10 10", "11 00"])
        p, xs = polys(f)
        x1, x2 = xs
        self.assertEqual((p[0]-x1*(1-x2)).simplify(), 0)
        self.assertEqual((p[1]-(1-x1)).simplify(), 0)
        self.assertEqual(polys_to_sd(p, xs), f)

    def test_random_state(self):
        ms = [4,5,2]
        x = random_state(ms)
        self.assertTrue(all(x[i]>=0 and x[i]<=ms[i] for i in range(len(ms))))
        n = 5
        x = random_boolean_state(n)
        self.assertTrue(all(xi==0 or xi==1 for xi in x))

    def test_random_map(self):
        n = 3
        f = random_boolean_map(n)
        self.assertTrue(all(f[x][i]==0 or f[x][i]==1 for x in f for i in range(n)))
        ms = [3,2]
        f = random_map(ms)
        self.assertTrue(all(f[x][i]>=0 and f[x][i]<=ms[i] for x in f for i in range(len(ms))))

    def test_generate_maps(self):
        ms = [2,1]
        self.assertEqual(len(list(generate_maps(ms))), 46656)
        n = 2
        self.assertEqual(len(list(generate_boolean_maps(n))), 256)

    def test_admissible(self):
        ms = [3,2,4,1]
        self.assertTrue(is_admissible((1,1,0,1,1,1,0,0,0,0), ms))
        self.assertFalse(is_admissible((1,1,0,1,1,0,1,0,0,0), ms))

    def test_stepwise_asymptotic(self):
        f1 = read_truth_table(["0 1", "1 2", "2 0"])
        f2 = read_truth_table(["00 12", "01 01", "02 00",
                               "10 20", "11 12", "12 12",
                               "20 02", "21 01", "22 11"])
        for f in [f1, f2]:
            self.assertFalse(is_stepwise(f))
            self.assertTrue(is_stepwise(to_stepwise(f)))
            self.assertFalse(is_asymptotic(f))
            self.assertTrue(is_asymptotic(to_asymptotic(f)))

    def test_expansive(self):
        f = read_truth_table(["00 00", "10 11"])
        self.assertTrue(is_expansive(f))
        f = read_truth_table(["00 00", "10 01"])
        self.assertFalse(is_expansive(f))

    def test_asynchronous(self):
        f = read_truth_table(["00 11", "10 11"])
        adf = {(0,0): set([(0,1),(1,0)]), (1,0): set([(1,1)])}
        self.assertEqual(sd_to_ad(f), adf)
        self.assertEqual(ad_to_sd(adf), f)

    def test_update(self):
        f = read_truth_table(["00 11", "01 01", "10 11", "11 00"])
        self.assertEqual(update(f, 1, (1,1)), (0,1))
        self.assertEqual(update(f, 2, (0,1)), (0,1))
        g = read_truth_table(["00 01", "01 01", "10 01", "11 10"])
        self.assertEqual(sequential(f, (2,1)), g)

    def test_trap_domain(self):
        f = read_truth_table(["00 11", "01 11", "10 11", "11 10"])
        self.assertTrue(is_trap_domain(f, [(1,0), (1,1)]))
        self.assertFalse(is_trap_domain(f, [(1,0), (0,0)]))

    def test_fixed_points(self):
        f = {x: (0,0) for x in boolean_states(2)}
        self.assertTrue(has_fixed_points(f))
        self.assertEqual(fixed_points(f), [(0,0)])
        f = read_truth_table(["00 11", "01 11", "10 11", "11 10"])
        self.assertFalse(has_fixed_points(f))

    def test_attractor(self):
        f = {x: (0,0) for x in boolean_states(2)}
        self.assertEqual(list(attractors(f)), [set([(0,0)])])
        self.assertEqual(list(attractors(f, synch=True)), [set([(0,0)])])
        f = read_truth_table(["00 11", "01 01", "10 11", "11 10"])
        self.assertEqual(list(attractors(f)), [set([(0,1)]), set([(1,0), (1,1)])])
        self.assertEqual(list(cyclic_attractors(f)), [set([(1,0), (1,1)])])
        self.assertEqual(list(attractive_cycles(f)), [[(1,0), (1,1)]])

    def test_mirror(self):
        ms = [3,2,4]
        f = random_map(ms)
        x, y, z = (1,0,2), (0,0,3), (1,0,3)
        f[x], f[y], f[z] = (0,1,4), (2,0,1), (2,1,3)
        self.assertFalse(is_mirror_pair(f, x, x))
        self.assertTrue(is_mirror_pair(f, x, y))
        self.assertFalse(is_mirror_pair(f, x, z))
        self.assertFalse(is_mirror_pair_fixed(f, x, y))
        fb = multi_to_boolean(f)
        xb, yb, zb = map(lambda c: to_boolean_vect(c, ms), [x, y, z])
        self.assertFalse(is_mirror_pair(fb, xb, xb))
        self.assertFalse(is_mirror_pair(fb, xb, xb, defn="cube"))
        self.assertFalse(is_mirror_pair(fb, xb, xb, defn="boolean"))
        self.assertTrue(is_mirror_pair(fb, xb, yb))
        self.assertTrue(is_mirror_pair(fb, xb, yb, defn="cube"))
        self.assertTrue(is_mirror_pair(fb, xb, yb, defn="boolean"))
        self.assertFalse(is_mirror_pair(fb, xb, zb))
        self.assertFalse(is_mirror_pair(fb, xb, zb, defn="cube"))
        self.assertFalse(is_mirror_pair(fb, xb, zb, defn="boolean"))
        self.assertFalse(is_mirror_pair_fixed(fb, xb, yb))
        self.assertFalse(is_mirror_pair_fixed(fb, xb, yb, defn="cube"))
        self.assertFalse(is_mirror_pair_fixed(fb, xb, yb, defn="boolean"))
        f[x], f[y] = x, y
        self.assertTrue(is_mirror_pair(f, x, y))
        self.assertTrue(is_mirror_pair_fixed(f, x, y))
        fb = multi_to_boolean(f)
        self.assertTrue(is_mirror_pair(fb, xb, yb))
        self.assertTrue(is_mirror_pair(fb, xb, yb, defn="cube"))
        self.assertTrue(is_mirror_pair(fb, xb, yb, defn="boolean"))
        self.assertTrue(is_mirror_pair_fixed(fb, xb, yb))
        self.assertTrue(is_mirror_pair_fixed(fb, xb, yb, defn="cube"))
        self.assertTrue(is_mirror_pair_fixed(fb, xb, yb, defn="boolean"))

    def test_mirror_circuits(self):
        ms = [2,3,2]
        for f in [random_map(ms) for i in range(5)]:
            mirror_pairs = [p for p in combinations(discrete_states(ms), 2) if is_mirror_pair(f, p[0], p[1])]
            if len(mirror_pairs)>0:
                lig = local_int_graph(f)
                self.assertTrue(any(len(lig[p[0]])>0 and len(lig[p[1]])>0 for p in mirror_pairs))

    def test_dist(self):
        self.assertTrue(dist_set((0,1), [(2,3),(0,1),(2,2)])==0)
        self.assertTrue(dist_set((0,1), [(2,3),(0,4),(2,1)])==2)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDin)
    unittest.TextTestRunner(verbosity=2).run(suite)
