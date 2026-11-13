"""Revision executor with profiled clocks, priority service and two-way tracking."""
from functools import partial
from itertools import pairwise
from math import inf, nextafter
from random import Random
from time import perf_counter_ns

from network import (ACT_BYTES, SWAP_SUCCESS, SWAP_MS, TRACK_BYTES,
                     Network, Packet, edge_key, seed_for)
from planner import BYTE_CAP, TELEMETRY_MS
from simulator import Runtime, State
from revision.controller import Calibrator, Controller, Selection
from revision.priority import PriorityNetwork

CUSTOM = {"batch_joint", "aware", "aware_nocost", "aware_independent"}


def tracking_bytes(catalog, plan):
    activation = ACT_BYTES * sum(len(r) - 1 for r in plan.activation)
    forward = sum(len(catalog.cpaths(a, b)[0]) - 1 for a, b in pairwise(plan.route))
    reverse = sum(len(catalog.cpaths(b, a)[0]) - 1 for a, b in pairwise(plan.route))
    return activation + TRACK_BYTES * (forward + reverse)


class RevisionRuntime(Runtime):
    def __init__(self, catalog, condition, policy, seed, requests, *, controller,
                 costs, clock="profiled", burn=0, record_outcomes=False):
        self.catalog, self.topo = catalog, catalog.topo
        self.condition, self.policy, self.seed = condition, policy, seed
        self.requests, self.controller = requests, controller
        self.costs, self.clock, self.burn = costs, clock, burn
        self.record_outcomes = record_outcomes
        self.fixed_planner_ms = None
        stop = requests[-1].offer_ms + 30
        outage = (seed % 5, requests[-1].offer_ms * .30, requests[-1].offer_ms * .74)
        net_type = PriorityNetwork if policy.endswith("_priority") else Network
        self.net = net_type(self.topo, outage if condition.outage else None)
        self.net.load_background(condition.load, seed_for(self.topo.name, seed), stop)
        self.free = dict(self.topo.slots)
        self.seen = {}
        self.controller_free = 0.
        self.rows = {}
        self.peak_used_slots = 0
        self.last_offer = None
        self.offer_gap = None
        self.feedback = 0.
        self.tracks = {}
        self.emitted = {}
        self.measurement_control_start = 0
        t = -8.
        while t <= stop:
            for node in self.topo.c_adj:
                self.net.at(t + .013 * node, self.telemetry, node)
            t += TELEMETRY_MS
        for request in requests:
            if request.number == burn:
                self.net.at(request.offer_ms, self.begin_measurement)
            self.net.at(request.offer_ms, self.on_offer, request)

    def begin_measurement(self, time):
        self.measurement_control_start = sum(v for k, v in self.net.bytes.items() if k != "background")

    def on_offer(self, time, request):
        if self.last_offer is not None:
            gap = time - self.last_offer
            self.offer_gap = gap if self.offer_gap is None else .9 * self.offer_gap + .1 * gap
        self.last_offer = time
        self.dispatch(time, request)

    def dispatch(self, time, r):
        if time < self.controller_free:
            self.net.at(self.controller_free, self.dispatch, r)
            return
        if time >= r.cutoff:
            self.rows[r.number] = self.row(r, "controller_timeout", False, time - r.offer_ms, 0.)
            return
        begin = perf_counter_ns()
        if self.policy in CUSTOM:
            selection = self.controller.decide(
                r.source, r.dest, r.cutoff - time, r.min_fidelity, self.condition.memory_scale,
                self.free, time, self.seen, self.policy,
                offer_rate=1 / self.offer_gap if self.offer_gap else 0., feedback=self.feedback,
            )
        else:
            helper = "qsp" if self.policy in {"qsp_priority", "track"} else self.policy
            choice = self.catalog.decide(
                r.source, r.dest, r.cutoff - time, r.min_fidelity, self.condition.memory_scale,
                self.free, time, self.seen, helper,
            )
            selection = Selection(choice.plan, choice.predicted, choice.predicted,
                                  helper, self.costs[helper], choice.reason)
            if self.policy == "track" and selection.plan is not None:
                b = self.controller.batches[r.source, r.dest]
                available = [p for p in b.plans if p.detours == 0
                             and all(self.free[v] >= n for v, n in p.qubits.items())
                             and tracking_bytes(self.catalog, p) <= BYTE_CAP]
                if not available:
                    selection = Selection(None, None, None, "qsp", self.costs["qsp"], "budget")
                else:
                    selection.plan = available[0]
                    # The legacy direct-collector forecast does not predict TRACK completion.
                    selection.raw = selection.predicted = None
        plan = selection.plan
        state = State(r, plan, time, dict(plan.qubits)) if plan is not None else None
        if state is not None and self.policy in {"replicate", "retry"}:
            self.select_backup(state)
        measured = (perf_counter_ns() - begin) / 1e6
        elapsed = measured if self.clock == "measured" else selection.cost_ms
        self.controller_free = time + elapsed
        if plan is None or self.controller_free >= r.cutoff:
            row = self.row(r, selection.reason or "controller_timeout", False,
                           time - r.offer_ms, measured, selection.predicted)
            self.rows[r.number] = row
        else:
            for node, needed in state.held.items():
                self.free[node] -= needed
                assert self.free[node] >= 0
            self.peak_used_slots = max(self.peak_used_slots,
                                      sum(self.topo.slots.values()) - sum(self.free.values()))
            row = self.row(r, "pending", True, time - r.offer_ms, measured, selection.predicted)
            row.update({"route": "-".join(map(str, plan.route)), "detours": plan.detours,
                        "selection_applied_ms": elapsed, "dispatch_ms": self.controller_free,
                        "reserved_byte_hops": self.reserved_traffic(state),
                        "outcomes": len(plan.controls)})
            self.rows[r.number] = row
            if self.policy == "track":
                self.tracks[r.number] = {"pending": {}, "sent": set()}
            for node, route in zip(plan.route, plan.activation, strict=True):
                self.net.at(self.controller_free, self.net.send,
                            Packet(route, ACT_BYTES, "activation", partial(self.activate, state, node)))
            self.net.at(nextafter(r.cutoff, inf), self.expire, state)
        row.update({"raw_predicted": selection.raw, "decision_mode": selection.mode,
                    "predicted_cost_ms": selection.cost_ms})
        if plan is not None and selection.raw is not None:
            row["calibrated_predicted"] = float(self.controller.calibrator.predict(
                selection.raw, self.condition.load, self.condition.memory_scale, len(plan.controls)))

    def row(self, r, *args, **kwargs):
        row = super().row(r, *args, **kwargs)
        row.update({"measurement": r.number >= self.burn, "nominal_load": self.condition.load,
                    "memory_scale": self.condition.memory_scale, "outcomes": None,
                    "raw_predicted": None, "calibrated_predicted": None,
                    "decision_mode": None, "predicted_cost_ms": None, "track_reverse_ms": None})
        return row

    def finish(self, time, state, status, quality=None):
        super().finish(time, state, status, quality)
        row = self.rows[state.request.number]
        # Only completed requests at current event time inform later decisions.
        if self.policy.startswith("aware") and row["predicted"] is not None:
            error = float(status == "qualified") - float(row["calibrated_predicted"])
            self.feedback = max(-.30, min(.30, .97 * self.feedback + .03 * error))

    def reserved_traffic(self, state):
        if self.policy == "track":
            return tracking_bytes(self.catalog, state.plan)
        return super().reserved_traffic(state)

    def herald(self, time, state, edge):
        if self.policy != "track":
            return super().herald(time, state, edge)
        if state.ended:
            return
        state.births[edge] = time
        q = state.plan.route
        if len(q) == 2:
            return self.deliver(time, state)
        if edge == edge_key(q[0], q[1]):
            self.send_track(time, state, 0, 1, 0)
        if edge == edge_key(q[-2], q[-1]):
            self.send_track(time, state, len(q) - 1, -1, 0)
        for node in edge:
            if node in q[1:-1] and node not in state.scheduled:
                i = q.index(node)
                if edge_key(q[i - 1], node) in state.births and edge_key(node, q[i + 1]) in state.births:
                    state.scheduled.add(node)
                    self.net.at(time + SWAP_MS, self.swap, state, node)

    def swap(self, time, state, node):
        if self.record_outcomes:
            self.emitted.setdefault(state.request.number, {})[node] = time
        if self.policy != "track":
            return super().swap(time, state, node)
        if state.ended:
            return
        rng = Random(seed_for(state.request.seed, "swap", node))
        ok, bits = rng.random() < SWAP_SUCCESS, rng.randrange(4)
        state.swaps[node] = time, ok, bits
        self.release(time, state, node)
        for direction in [1, -1]:
            self.forward_track(time, state, state.plan.route.index(node), direction)

    def send_track(self, time, state, index, direction, bits):
        q = state.plan.route
        route = self.catalog.cpaths(q[index], q[index + direction])[0]
        callback = partial(self.receive_track, state, index + direction, direction, bits)
        self.net.send(time, Packet(route, TRACK_BYTES, "tracking", callback))

    def receive_track(self, state, index, direction, bits, time):
        if state.ended:
            return
        q = state.plan.route
        if index == len(q) - 1 and direction == 1:
            state.received[q[-1]] = bits
            self.deliver(time, state)
        elif index == 0 and direction == -1:
            self.rows[state.request.number]["track_reverse_ms"] = time - state.request.offer_ms
        else:
            self.tracks[state.request.number]["pending"][index, direction] = bits
            self.forward_track(time, state, index, direction)

    def forward_track(self, time, state, index, direction):
        if state.ended:
            return
        tracks = self.tracks[state.request.number]
        key = index, direction
        node = state.plan.route[index]
        if key in tracks["sent"] or key not in tracks["pending"] or node not in state.swaps:
            return
        tracks["sent"].add(key)
        _, ok, bits = state.swaps[node]
        if ok:
            self.send_track(time, state, index, direction, bits ^ tracks["pending"][key])
