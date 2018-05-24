#!/usr/bin/env python

"""Tests for dinpy."""

import unittest

from dinpy.input_din import read_truth_table, random_map
from dinpy.din import boolean_states, discrete_states, to_stepwise, diff_inds, sign, is_admissible, neigh
from dinpy.interaction_graphs import global_int_graph, local_int_graph, global_circuits, local_circuits, nu_int_graph, local_int_graph_state, local_int_graph_attr_state
from dinpy.multi_to_boolean import multi_to_boolean, multi_to_boolean_adm, multi_level_to_bool, to_boolean_vect, bool_vars_to_multi, to_multi


class TestIntGraph(unittest.TestCase):
    def test_int_graph(self):
        f1 = read_truth_table(["00 01", "01 00", "11 01", "10 00"])
        f2 = read_truth_table(["00 11", "01 10", "11 11", "10 10"])
        lig1 = local_int_graph(f1)
        lig2 = local_int_graph(f2)
        lg = {(0,0): [(1,2,-1), (2,2,-1)], (0,1): [(1,2,1), (2,2,-1)],
              (1,0): [(1,2,-1), (2,2,+1)], (1,1): [(1,2,1), (2,2,+1)]}
        self.assertEqual(lig1, lig2)
        self.assertEqual(lig1, lg)
        grg1 = global_int_graph(f1)
        grg2 = global_int_graph(f2)
        gg = [(1,2,-1), (1,2,1), (2,2,-1), (2,2,1)]
        self.assertEqual(grg1, grg2)
        self.assertEqual(grg1, gg)

    def test_circuits(self):
        f1 = read_truth_table(["00 00", "01 10", "10 01", "11 10"])
        lcs1 = {(0,0): [([1,2], +1)], (0,1): [], (1,0): [([1,2], +1), ([2], -1)], (1,1): [([2], -1)]}
        lcs1n = {(0,0): [], (0,1): [], (1,0): [([2], -1)], (1,1): [([2], -1)]}
        lcs1p = {(0,0): [([1,2], +1)], (0,1): [], (1,0): [([1,2], +1)], (1,1): []}
        self.assertEqual(lcs1, local_circuits(f1))
        self.assertEqual(lcs1n, local_circuits(f1, sign=-1))
        self.assertEqual(lcs1p, local_circuits(f1, sign=+1))
        gcs1 = [([1,2], +1), ([2], -1)]
        gcs1n = [([2], -1)]
        gcs1p = [([1,2], +1)]
        self.assertEqual(gcs1, global_circuits(f1))
        self.assertEqual(gcs1n, global_circuits(f1, sign=-1))
        self.assertEqual(gcs1p, global_circuits(f1, sign=+1))
        f2 = read_truth_table(["00 11", "01 01", "10 00", "11 00"])
        lcs2 = {(0,0): [([1,2], +1), ([1], -1)], (0,1): [([1,2], +1)], (1,0): [([1], -1)], (1,1): []}
        lcs2n = {(0,0): [([1], -1)], (0,1): [], (1,0): [([1], -1)], (1,1): []}
        lcs2p = {(0,0): [([1,2], +1)], (0,1): [([1,2], +1)], (1,0): [], (1,1): []}
        self.assertEqual(lcs2, local_circuits(f2))
        self.assertEqual(lcs2n, local_circuits(f2, sign=-1))
        self.assertEqual(lcs2p, local_circuits(f2, sign=+1))
        gcs2 = [([1], -1), ([1,2], +1)]
        gcs2n = [([1], -1)]
        gcs2p = [([1,2], +1)]
        self.assertEqual(gcs2, global_circuits(f2))
        self.assertEqual(gcs2n, global_circuits(f2, sign=-1))
        self.assertEqual(gcs2p, global_circuits(f2, sign=+1))

    def test_non_usual(self):
        f = read_truth_table(["00 21", "01 02", "02 20",
                              "10 20", "11 00", "12 02",
                              "20 20", "21 10", "22 01"])
        nug = nu_int_graph(f)
        self.assertEqual(nug[(1,2), (0,1)], [(1,1,-1), (1,2,1), (2,2,1)])
        self.assertEqual(nug[(1,2), (1,0)], [(2,2,1)])
        self.assertEqual(nug[(0,1), (1,0)], [(1,2,-1), (2,1,-1)])
        gnug = [(1,1,-1), (1,2,-1), (1,2,1), (2,1,-1), (2,1,1), (2,2,-1), (2,2,1)]
        self.assertEqual(global_int_graph(f, graph=nu_int_graph), gnug)
        lcnu = local_circuits(f, graph=nu_int_graph)
        self.assertEqual(lcnu[(1,2), (0,1)], [([1], -1), ([2], +1)])
        self.assertEqual(lcnu[(0,1), (1,0)], [([1,2], 1)])
        gcnu = global_circuits(f, graph=nu_int_graph)
        self.assertEqual(gcnu, [([1], -1), ([1,2], -1), ([1,2], +1), ([2], -1), ([2], +1)])

    def test_non_usual_conversion(self):
        ms = [2,1,3]
        n = sum(ms)
        f = random_map(ms)
        fb = multi_to_boolean_adm(f)
        Fb = multi_to_boolean(f)
        nug = nu_int_graph(f)
        mltb, btml = multi_level_to_bool(ms), bool_vars_to_multi(ms)

        for x, y in nug:
            if len(nug[(x,y)])>0:
                bx, by = to_boolean_vect(x, ms), to_boolean_vect(y, ms)
                I = [i for i in diff_inds(bx, by) if is_admissible(neigh(bx, i), ms)]
                gxfb = local_int_graph_state(fb, bx, ms=[1]*n, I=I)
                gxFb = local_int_graph_state(Fb, bx, ms=[1]*n, I=I)
                self.assertEqual(gxfb, gxFb)
                self.assertTrue([e in gxFb for e in gxfb])

                # edge in non-usual -> edge in Gfb(x)
                for j, i, s in nug[(x,y)]:
                    ei, ej = sign(y[i-1]-x[i-1]), sign(y[j-1]-x[j-1])
                    self.assertTrue((mltb[(j,int(x[j-1]+(ej+1)/2))], mltb[(i,int(x[i-1]+(ei+1)/2))], s) in gxfb)

                # circuit in Gf(x,y) -> circuit in Gfb(x)
                lcxy = local_circuits(f, graph=nu_int_graph, at=(x, y))
                lcbxby = local_circuits(fb, graph=local_int_graph_state, I=I, at=bx)
                if len(lcxy)>0:
                    slcxy, slcbxby = [s for _, s in lcxy], [s for _, s in lcbxby]
                    self.assertTrue(all(s in slcbxby for s in slcxy))

        for x in f:
            bx = to_boolean_vect(x, ms)
            lcbxby = local_circuits(fb, graph=local_int_graph_state, at=bx)
            # circuit in Gfb(x) -> circuit in non=-usual
            if len(lcbxby)>0:
                for c, s in lcbxby:
                    inds = [btml[ci] for ci in c]
                    inds0 = [i for i, _ in inds]
                    if len(list(set(inds0)))==len(inds0):
                        by = bx
                        for i in c: by = neigh(by, i-1)
                        y = to_multi(by, ms)
                        lcxy = local_circuits(f, graph=nu_int_graph, at=(x, y))
                        slcxy = [s for _, s in lcxy]
                        self.assertTrue(s in slcxy)

        f = read_truth_table(["00 10", "01 10", "02 01", "10 21", "11 11", "12 01", "20 21", "21 12", "22 12"])
        fb = multi_to_boolean_adm(f)
        lc = local_circuits(f, graph=nu_int_graph)
        lcb = local_circuits(fb, graph=local_int_graph_state)
        self.assertTrue(len(lc[x])==0 for x in lc)
        self.assertTrue(any(len(lcb[x])>0 for x in lcb))

    def test_stepwise_rg(self):
        ms = [3,2]
        f = random_map(ms)
        uf = to_stepwise(f)
        lig, ligu = local_int_graph(f), local_int_graph(uf)
        self.assertTrue(all(e in lig[x] for x in f for e in ligu[x] if e[0]!=e[1]))
        f = {x: (1,2) for x in discrete_states(ms)}
        f[(0,1)], f[(0,2)] = (2,2), (2,2)
        uf = to_stepwise(f)
        lig, ligu = local_int_graph(f), local_int_graph(uf)
        self.assertFalse(all(e in lig[x] for x in f for e in ligu[x] if e[0]==e[1]))
        self.assertFalse(all(e in ligu[x] for x in f for e in lig[x] if e[0]==e[1]))

    def test_ig_attr(self):
        ms = [3,2]
        f = to_stepwise(random_map(ms))
        lig = local_int_graph(f)
        lig1 = local_int_graph(f, graph=local_int_graph_attr_state)
        self.assertTrue(all(e in lig[x] for x in f for e in lig1[x]))
        f = {(0,0): (1,2), (0,1): (1,1), (0,2): (1,0),
             (1,0): (1,2), (1,1): (0,2), (1,2): (1,1)}
        lig = local_int_graph(f)
        lig1 = local_int_graph(f, graph=local_int_graph_attr_state)
        self.assertFalse(all(e in lig[x] for x in f for e in lig1[x]))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntGraph)
    unittest.TextTestRunner(verbosity=2).run(suite)
