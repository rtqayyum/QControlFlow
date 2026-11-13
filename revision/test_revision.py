import unittest
from unittest.mock import patch
from collections import Counter
from itertools import pairwise
from math import log

import numpy as np

from campaign import choose_pairs
from network import ClassicalLink, Network, Packet, QuantumLink, Topology, adjacency, make_topology
from planner import Catalog
from revision.controller import Batch, Calibrator, Controller, coupled_arrivals
from revision.priority import PriorityNetwork
from revision.runtime import RevisionRuntime, tracking_bytes
from simulator import Condition, Request, offer_requests


def line_topology(n=4):
    edges = {(i, i + 1) for i in range(n - 1)}
    directed = {e for a, b in edges for e in [(a, b), (b, a)]}
    return Topology("unit-line", {e: QuantumLink(.2, 1., .999, 1000.) for e in edges},
                    {e: ClassicalLink(1000., .001, 0., 0) for e in directed},
                    adjacency(n, directed), adjacency(n, directed), {i: 8 for i in range(n)})


class RevisionTests(unittest.TestCase):
    def test_common_link_outcomes_serialize(self):
        topo = line_topology(2)
        topo.c[0, 1] = ClassicalLink(8., 0., 0., 0)
        got = coupled_arrivals(topo, [(0, 1), (0, 1)], np.zeros((2, 1)), {(0, 1): np.zeros(1)})
        np.testing.assert_allclose(got[:, 0], [.096, .192])

    def test_disjoint_outcomes_do_not_interact(self):
        topo = line_topology(3)
        got = coupled_arrivals(topo, [(0, 1), (2, 1)], np.zeros((2, 2)),
                               {(0, 1): np.zeros(2), (2, 1): np.zeros(2)})
        np.testing.assert_allclose(got, .001768)

    def test_priority_is_nonpreemptive(self):
        topo = line_topology(2)
        topo.c[0, 1] = ClassicalLink(8., 0., 0., 0)
        net = PriorityNetwork(topo)
        order = []
        net.at(0., net.send, Packet((0, 1), 1500, "background", lambda t: order.append(("first", t))))
        net.at(.01, net.send, Packet((0, 1), 1500, "background", lambda t: order.append(("second", t))))
        net.at(.02, net.send, Packet((0, 1), 96, "outcome", lambda t: order.append(("control", t))))
        net.run()
        self.assertEqual([x[0] for x in order], ["first", "control", "second"])
        np.testing.assert_allclose([x[1] for x in order], [1.5, 1.596, 3.096])

    def test_priority_tail_drop_uses_shared_finite_capacity(self):
        topo = line_topology(2)
        net = PriorityNetwork(topo)
        for i in range(17):
            net.at(0., net.send, Packet((0, 1), 1500, "background"))
        net.run()
        self.assertEqual(net.drops["background"], 1)
        self.assertEqual(net.bg_delivered, 16)

    def test_isotonic_calibrator_is_monotone_and_shrunk(self):
        values = Calibrator.monotone([.1, .7, .4, .9], [1, 1, 1, 1])
        np.testing.assert_allclose(values, [.1, .55, .55, .9])
        rows = [{"admitted": True, "raw_predicted": .9, "nominal_load": .73,
                 "memory_scale": 1., "outcomes": 2, "status": "qualified" if i < 40 else "fidelity"}
                for i in range(100)]
        calibrator = Calibrator.fit(rows)
        self.assertLess(calibrator.predict(.9, .73, 1., 2), .55)

    @classmethod
    def setUpClass(cls):
        cls.topo = make_topology("synthetic-24-1")
        cls.catalog = Catalog(cls.topo, .73)
        cls.pairs = choose_pairs(cls.topo)[:3]
        cls.catalog.warm(cls.pairs)
        cls.controller = Controller(cls.catalog, {"base": .03, "full": .10})

    def test_independent_batch_matches_direct_sampled_score(self):
        for pair, b in self.controller.batches.items():
            for minimum, scale, remaining in [(.84, 1., 16.), (.91, .6, 7.), (.88, 1., 24.)]:
                shifts = b.shifts(0., {})
                scores = b.score(remaining, minimum, scale, shifts, coupled=False)
                for i, p in enumerate(b.plans):
                    frame = p.ready if not p.controls else np.maximum(p.ready, p.arrivals.max(axis=0))
                    loss = (p.pre_exposure + p.endpoint_rate * (frame - p.ready)) / scale
                    direct = np.mean(p.valid & (frame <= remaining)
                                     & (p.log_raw - loss >= log((4 * minimum - 1) / 3)))
                    self.assertEqual(scores[i], direct)

    def test_coupling_cannot_accelerate_frozen_background(self):
        for b in self.controller.batches.values():
            mask = np.isfinite(b.independent)
            self.assertTrue(np.all(b.coupled[mask] >= b.independent[mask] - 1e-10))

    def test_subtracting_cost_can_only_remove_deadline_successes(self):
        for b in self.controller.batches.values():
            shifts = b.shifts(0., {})
            early = b.score(16., .84, 1., shifts)
            late = b.score(8., .84, 1., shifts)
            self.assertTrue(np.all(late <= early))

    def test_high_opportunity_cost_skips_search(self):
        s, d = self.pairs[0]
        result = self.controller.decide(s, d, 24., .84, 1., self.topo.slots, 0., {}, "aware", 1e6)
        self.assertEqual(result.mode, "base")
        self.assertEqual(result.cost_ms, .03)

    def test_two_way_tracking_uses_both_directions_and_same_frame(self):
        topo = line_topology()
        catalog = Catalog(topo, 0.)
        catalog.warm([(0, 3)])
        controller = Controller(catalog)
        condition = Condition("unit", 0., 1., 1.)
        r = [Request(0, 0, 3, 1., 10., .8, 912)]
        costs = {"qsp": .02}
        with patch("simulator.SWAP_SUCCESS", 1.), patch("revision.runtime.SWAP_SUCCESS", 1.):
            direct, _ = RevisionRuntime(catalog, condition, "qsp", 1, r,
                                        controller=controller, costs=costs).run()
            tracked, counters = RevisionRuntime(catalog, condition, "track", 1, r,
                                                controller=controller, costs=costs).run()
        self.assertEqual(direct[0]["status"], "qualified")
        self.assertEqual(tracked[0]["status"], "qualified")
        self.assertEqual(direct[0]["frame"], tracked[0]["frame"])
        self.assertEqual(tracked[0]["reserved_byte_hops"], tracking_bytes(catalog, catalog.table[0, 3][0]))
        self.assertGreaterEqual(counters["bytes_by_type"]["tracking"], 4 * 96)

    def test_revision_policies_finish_with_conserved_slots(self):
        condition = Condition("unit", .73, 1., 1.45)
        requests = offer_requests(self.topo, self.pairs, condition, 8131, 12)
        costs = {"qsp": .03, "fast_sequential": .05, "fast_joint": .08}
        for policy in ["qsp", "fast_sequential", "fast_joint", "batch_joint", "aware",
                       "aware_nocost", "aware_independent", "qsp_priority", "track"]:
            runtime = RevisionRuntime(self.catalog, condition, policy, 8131, requests,
                                      controller=self.controller, costs=costs)
            rows, _ = runtime.run()
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(r["status"] != "pending" for r in rows))
            self.assertTrue(all(r["reserved_byte_hops"] <= 3000 for r in rows if r["admitted"]))
            self.assertEqual(runtime.free, self.topo.slots)


if __name__ == "__main__":
    unittest.main()
