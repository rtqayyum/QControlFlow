import unittest
from collections import Counter
from itertools import product
from math import exp

from compare_budget_matched import extra_hops, joint_choices, reserved, sequential_choices
from pilot_model import (
    ATTEMPTS,
    OUTCOME_ROUTES,
    QUANTUM,
    QUEUE_PACKETS,
    Outcome,
    Packet,
    PacketNetwork,
    Plan,
    Workload,
    fidelity,
    make_workload,
    plan_catalog,
    simulate,
    visibility,
)
from run_pilot import Condition, Metrics, choose, seed_lists


def fixed_tape(*, right_link_attempt=1, background=(), failing_swap=None):
    generation = {edge: (0.0,) * ATTEMPTS for edge in QUANTUM}
    generation["R2", "B"] = (
        (1.0,) * (right_link_attempt - 1) + (0.0,) * (ATTEMPTS + 1 - right_link_attempt)
    )
    swaps = {node: (1.0 if node == failing_swap else 0.0, i + 1)
             for i, node in enumerate(OUTCOME_ROUTES)}
    return Workload(1, generation, swaps, tuple(background))


class PilotTests(unittest.TestCase):
    def test_visibility_time_budget_matches_direct_fidelity(self):
        outcome = Outcome(None, 0.0, 5.0, visibility(0.94), 0.0, 2 / 60, 0, 0, 0)
        self.assertAlmostEqual(outcome.wait_allowance_ms(0.85, 1.0), 4.19285827125476)
        self.assertAlmostEqual(outcome.delivered_fidelity(1.0), fidelity(0.92 * exp(-5 / 30)))
        self.assertEqual(outcome.verdict(8.0, 0.85, 1.0), "fidelity")

    def test_fifo_serialization_and_tail_drop(self):
        net = PacketNetwork()
        arrivals = []
        for i in range(QUEUE_PACKETS + 1):
            net.at(0.0, net.send, Packet(str(i), ("R2", "B"), 1500, arrivals.append))
        net.run()
        self.assertEqual(len(arrivals), QUEUE_PACKETS)
        self.assertEqual(net.control_drops, 1)
        self.assertEqual(net.control_bytes_hops, 1500 * QUEUE_PACKETS)
        self.assertAlmostEqual(arrivals[0], 1.68)
        self.assertAlmostEqual(arrivals[1] - arrivals[0], 1.5)

    def test_local_swap_runs_before_distant_link_is_ready(self):
        trace = []
        result = simulate(Plan("A", (0, 0)), fixed_tape(right_link_attempt=3), 0.10, trace)
        swap = next(e["time_ms"] for e in trace if e["event"] == "swap_complete" and e["node"] == "R1")
        distant = next(e["time_ms"] for e in trace if e["event"] == "elementary_pair_ready"
                       and e["edge"] == ("R2", "B"))
        self.assertLess(swap, distant)
        self.assertEqual(result.frame, 1 ^ 2)
        outcome_arrivals = [e["time_ms"] for e in trace if e["event"] == "packet_arrival"
                            and e["packet"].startswith("outcome:")]
        self.assertEqual(len(outcome_arrivals), 2)
        self.assertEqual(result.frame_ready_ms, max(outcome_arrivals))

    def test_same_quantum_path_rescued_by_control_reroute(self):
        tape = fixed_tape(background=[(1.5, ("R2", "B"))] * 4)
        original = simulate(Plan("A", (0, 0)), tape, 0.10)
        alternate = simulate(Plan("A", (1, 1)), tape, 0.10)
        self.assertEqual(original.ready_ms, alternate.ready_ms)
        self.assertEqual(original.pre_gate_exposure, alternate.pre_gate_exposure)
        self.assertEqual(original.frame, alternate.frame)
        self.assertEqual(original.verdict(10.0, 0.85, 1.0), "fidelity")
        for choices in ((0, 1), (1, 0)):
            with self.subTest(choices=choices):
                single = simulate(Plan("A", choices), tape, 0.10)
                self.assertEqual(single.verdict(10.0, 0.85, 1.0), "fidelity")
        self.assertEqual(alternate.verdict(10.0, 0.85, 1.0), "qualified")

    def test_failed_swap_cannot_complete_request(self):
        result = simulate(Plan("A", (0, 0)), fixed_tape(failing_swap="R2"), 0.10)
        self.assertEqual(result.verdict(10.0, 0.85, 1.0), "swap")
        self.assertIsNone(result.frame_ready_ms)

    def test_missing_outcome_is_a_failure(self):
        tape = fixed_tape(background=[(1.79, ("R2", "B"))] * QUEUE_PACKETS)
        result = simulate(Plan("A", (0, 0)), tape, 0.10)
        self.assertEqual(result.verdict(10.0, 0.85, 1.0), "outcome_loss")

    def test_planning_is_in_the_request_clock(self):
        plan = Plan("B", (0,))
        tape = fixed_tape()
        start = simulate(plan, tape, 0.10)
        delayed = simulate(plan, tape, 1.10)
        self.assertAlmostEqual(delayed.frame_ready_ms - start.frame_ready_ms, 1.0)
        self.assertAlmostEqual(start.delivered_fidelity(1.0), delayed.delivered_fidelity(1.0))

    def test_holdout_is_disjoint_and_workload_is_reproducible(self):
        cal, test = seed_lists(20261112, 50, 100)
        self.assertFalse(set(cal) & set(test))
        self.assertEqual(make_workload(test[0], 0.9), make_workload(test[0], 0.9))
        self.assertEqual(make_workload(test[0], 0.9).generation, make_workload(test[0], 0.25).generation)

    def test_joint_choice_respects_cap_and_can_change_quantum_path(self):
        plans = plan_catalog()
        successes = {"A-00": 40, "A-01": 75, "A-10": 42, "A-11": 88, "B-0": 65, "B-1": 67}
        metrics = {}
        for plan in plans:
            good = successes[plan.key]
            counts = dict(Counter(qualified=good, deadline=100 - good))
            metrics[plan.key] = Metrics(100, counts, 8.0, 0.87, 9.0, 250.0, 0)
        condition = Condition(0.9, 1.0, 0.9)
        wide = choose(plans, metrics, condition, 576)
        narrow = choose(plans, metrics, condition, 480)
        self.assertEqual(wide["sequential"].key, "B-1")
        self.assertEqual(wide["joint"].key, "A-11")
        self.assertEqual(narrow["joint"].key, "B-1")


    def test_exact_budget_optimizer_matches_brute_force(self):
        plans = plan_catalog()
        calibration = [
            {p.key: (i + 3) * (j + 2) % 11 for j, p in enumerate(plans)}
            for i in range(3)
        ]
        brute = {}
        for combination in product(plans, repeat=len(calibration)):
            key = (sum(p.quantum_path == "A" for p in combination),
                   sum(extra_hops(p) for p in combination))
            score = sum(c[p.key] for c, p in zip(calibration, combination, strict=True))
            brute[key] = max(brute.get(key, 0), score)
        winners = joint_choices(calibration)
        self.assertEqual(set(winners), set(brute))
        for key, winner in winners.items():
            self.assertEqual(sum(c[p] for c, p in zip(calibration, winner, strict=True)), brute[key])
            self.assertEqual(reserved(winner), reserved(sequential_choices(calibration, *key)))


if __name__ == "__main__":
    unittest.main()
