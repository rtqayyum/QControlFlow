"""Event execution for overlapping entanglement requests and shared control links."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import pairwise
from math import exp, inf, nextafter
from random import Random
from time import perf_counter_ns

from network import (
    ACT_BYTES, ATTEMPTS, SWAP_MS, SWAP_SUCCESS, TRACK_BYTES, Edge, Network,
    Packet, PathNodes, Topology, edge_key, residual, seed_for,
)
from planner import BYTE_CAP, TELEMETRY_MS, Catalog, Forecast

ACK_BYTES = 32


def fidelity_from_exposure(raw_visibility: float, exposure: float, dephasing: bool = False) -> float:
    retention = exp(-exposure)
    if dephasing:
        return (1 + raw_visibility * (1 + 2 * retention)) / 4
    return (1 + 3 * raw_visibility * retention) / 4


@dataclass(frozen=True)
class Condition:
    name: str
    load: float
    memory_scale: float
    mean_gap_ms: float
    outage: bool = False
    dephasing: bool = False
    cold: bool = False


@dataclass(frozen=True)
class Request:
    number: int
    source: int
    dest: int
    offer_ms: float
    deadline_ms: float
    min_fidelity: float
    seed: int

    @property
    def cutoff(self) -> float:
        return self.offer_ms + self.deadline_ms


def offer_requests(topo: Topology, pairs: list[tuple[int, int]], condition: Condition,
                   seed: int, n: int) -> list[Request]:
    rng = Random(seed_for("requests", topo.name, seed))
    time = 0.0
    result = []
    for number in range(n):
        time += rng.expovariate(1 / condition.mean_gap_ms)
        source, dest = rng.choice(pairs)
        deadline, fidelity = rng.choice(((16.0, 0.84), (24.0, 0.88), (30.0, 0.91)))
        result.append(Request(number, source, dest, time, deadline, fidelity,
                              seed_for("physical", topo.name, seed, number)))
    return result


@dataclass
class State:
    request: Request
    plan: Forecast
    book: float
    held: dict[int, int]
    active: set[int] = field(default_factory=set)
    started: set[Edge] = field(default_factory=set)
    births: dict[Edge, float] = field(default_factory=dict)
    scheduled: set[int] = field(default_factory=set)
    swaps: dict[int, tuple[float, bool, int]] = field(default_factory=dict)
    received: dict[int, int] = field(default_factory=dict)
    chain_frames: dict[int, int] = field(default_factory=dict)
    chain_sent: set[int] = field(default_factory=set)
    bad_generation: bool = False
    acked: bool = False
    ended: bool = False
    memory_time: float = 0.0
    extra_relay: int | None = None
    ack_route: PathNodes | None = None


class Runtime:
    def __init__(self, catalog: Catalog, condition: Condition, policy: str, seed: int,
                 requests: list[Request], fixed_planner_ms: float | None = None) -> None:
        self.catalog = catalog
        self.topo = catalog.topo
        self.condition = condition
        self.policy = policy
        self.seed = seed
        self.requests = requests
        self.fixed_planner_ms = fixed_planner_ms
        stop = requests[-1].offer_ms + 30
        outage = (seed % 5, requests[-1].offer_ms * 0.30, requests[-1].offer_ms * 0.74)
        self.net = Network(self.topo, outage if condition.outage else None)
        self.net.load_background(condition.load, seed_for(self.topo.name, seed), stop)
        self.free = dict(self.topo.slots)
        self.seen: dict[Edge, tuple[float, float]] = {}
        self.controller_free = 0.0
        self.rows: dict[int, dict] = {}
        self.peak_used_slots = 0
        # Control policies see identical reporting infrastructure and pay its traffic.
        t = -8.0
        while t <= stop:
            for node in self.topo.c_adj:
                self.net.at(t + 0.013 * node, self.telemetry, node)
            t += TELEMETRY_MS
        for request in requests:
            self.net.at(request.offer_ms, self.dispatch, request)

    def telemetry(self, time: float, node: int) -> None:
        snapshot = {edge: residual(self.net.queues[edge], time)
                    for edge in ((node, dst) for dst in self.topo.c_adj[node])}

        def receive(_: float) -> None:
            for edge, work in snapshot.items():
                if edge not in self.seen or time > self.seen[edge][0]:
                    self.seen[edge] = (time, work)

        route = self.catalog.cpaths(node, self.topo.hub)[0]
        self.net.send(time, Packet(route, 48 + 8 * len(snapshot), "telemetry", receive))

    def dispatch(self, time: float, r: Request) -> None:
        if time < self.controller_free:
            self.net.at(self.controller_free, self.dispatch, r)
            return
        if time >= r.cutoff:
            self.rows[r.number] = self.row(r, "controller_timeout", False, time - r.offer_ms, 0.0)
            return
        begin = perf_counter_ns()
        choice = self.catalog.decide(
            r.source, r.dest, r.cutoff - time, r.min_fidelity, self.condition.memory_scale,
            self.free, time, self.seen, self.policy, self.condition.cold,
        )
        plan = choice.plan
        state = State(r, plan, time, dict(plan.qubits)) if plan is not None else None
        if state is not None and self.policy in ("replicate", "retry"):
            self.select_backup(state)
        measured = (perf_counter_ns() - begin) / 1e6
        elapsed = measured if self.fixed_planner_ms is None else self.fixed_planner_ms
        self.controller_free = time + elapsed
        if plan is None or self.controller_free >= r.cutoff:
            reason = choice.reason or "controller_timeout"
            self.rows[r.number] = self.row(r, reason, False, time - r.offer_ms, measured,
                                           choice.predicted)
            return

        for node, count in state.held.items():
            self.free[node] -= count
            assert self.free[node] >= 0
        used = sum(self.topo.slots.values()) - sum(self.free.values())
        self.peak_used_slots = max(self.peak_used_slots, used)
        self.rows[r.number] = self.row(r, "pending", True, time - r.offer_ms, measured,
                                       choice.predicted)
        self.rows[r.number].update({
            "route": "-".join(map(str, plan.route)), "detours": plan.detours,
            "selection_applied_ms": elapsed, "dispatch_ms": self.controller_free,
            "reserved_byte_hops": self.reserved_traffic(state),
        })
        for node, route in zip(plan.route, plan.activation, strict=True):
            packet = Packet(route, ACT_BYTES, "activation", partial(self.activate, state, node))
            self.net.at(self.controller_free, self.net.send, packet)
        self.net.at(nextafter(r.cutoff, inf), self.expire, state)

    def reserved_traffic(self, s: State) -> int:
        if self.policy == "chain":
            return (
                ACT_BYTES * sum(len(route) - 1 for route in s.plan.activation)
                + TRACK_BYTES * sum(len(route) - 1 for route in s.plan.chain)
            )
        if s.extra_relay is None:
            return s.plan.risk_bytes
        index = s.plan.route.index(s.extra_relay) - 1
        assert s.ack_route is not None
        return (
            s.plan.risk_bytes + TRACK_BYTES * (len(s.plan.backups[index]) - 1)
            + ACK_BYTES * (len(s.ack_route) - 1)
        )

    def select_backup(self, s: State) -> None:
        candidates = sorted(range(len(s.plan.controls)),
                            key=lambda i: -len(s.plan.controls[i]))
        for i in candidates:
            backup = s.plan.backups[i]
            node = s.plan.route[i + 1]
            ack_route = self.catalog.cpaths(s.request.dest, node)[0]
            cost = TRACK_BYTES * (len(backup) - 1) + ACK_BYTES * (len(ack_route) - 1)
            if backup != s.plan.controls[i] and s.plan.risk_bytes + cost <= BYTE_CAP:
                s.extra_relay, s.ack_route = node, ack_route
                return

    def activate(self, s: State, node: int, time: float) -> None:
        if s.ended:
            return
        s.active.add(node)
        for u, v in pairwise(s.plan.route):
            edge = edge_key(u, v)
            if edge in s.started or not (u in s.active and v in s.active):
                continue
            s.started.add(edge)
            link = self.topo.q[edge]
            rng = Random(seed_for(s.request.seed, "edge", edge))
            for trial in range(1, ATTEMPTS + 1):
                if rng.random() < link.p:
                    self.net.at(time + trial * link.ms, self.herald, s, edge)
                    break
            else:
                self.net.at(time + ATTEMPTS * link.ms, self.exhausted, s)

    def exhausted(self, time: float, s: State) -> None:
        if not s.ended:
            s.bad_generation = True

    def herald(self, time: float, s: State, edge: Edge) -> None:
        if s.ended:
            return
        s.births[edge] = time
        q = s.plan.route
        if len(q) == 2:
            self.deliver(time, s)
            return
        if self.policy == "chain" and edge == edge_key(q[0], q[1]):
            self.send_chain(time, s, 0, 0)
        for node in edge:
            if node in q[1:-1] and node not in s.scheduled:
                i = q.index(node)
                if edge_key(q[i - 1], node) in s.births and edge_key(node, q[i + 1]) in s.births:
                    s.scheduled.add(node)
                    self.net.at(time + SWAP_MS, self.swap, s, node)

    def swap(self, time: float, s: State, node: int) -> None:
        if s.ended:
            return
        rng = Random(seed_for(s.request.seed, "swap", node))
        ok, bits = rng.random() < SWAP_SUCCESS, rng.randrange(4)
        s.swaps[node] = (time, ok, bits)
        self.release(time, s, node)
        i = s.plan.route.index(node)
        if self.policy == "chain":
            self.forward_chain(time, s, i)
            return
        primary = s.plan.controls[i - 1]
        packet = Packet(primary, TRACK_BYTES, "outcome", partial(self.receive, s, node, ok, bits))
        self.net.send(time, packet)
        if node == s.extra_relay:
            backup = Packet(s.plan.backups[i - 1], TRACK_BYTES, "copy", packet.finish)
            if self.policy == "replicate":
                self.net.send(time, backup)
            else:
                self.net.at(time + 2.2, self.retry, s, backup)

    def retry(self, time: float, s: State, packet: Packet) -> None:
        if not s.acked and time < s.request.cutoff:
            self.net.send(time, packet)

    def receive(self, s: State, node: int, ok: bool, bits: int, time: float) -> None:
        if s.ended or node in s.received:
            return
        if node == s.extra_relay:
            def ack(_: float) -> None:
                s.acked = True
            self.net.send(time, Packet(s.ack_route, ACK_BYTES, "ack", ack))
        s.received[node] = bits
        if not ok:
            self.finish(time, s, "swap")
        elif len(s.received) == len(s.plan.controls):
            self.deliver(time, s)

    def send_chain(self, time: float, s: State, i: int, bits: int) -> None:
        packet = Packet(s.plan.chain[i], TRACK_BYTES, "tracking",
                        partial(self.receive_chain, s, i + 1, bits))
        self.net.send(time, packet)

    def receive_chain(self, s: State, i: int, bits: int, time: float) -> None:
        if s.ended:
            return
        if i == len(s.plan.route) - 1:
            s.received[s.request.dest] = bits
            self.deliver(time, s)
        else:
            s.chain_frames[i] = bits
            self.forward_chain(time, s, i)

    def forward_chain(self, time: float, s: State, i: int) -> None:
        node = s.plan.route[i]
        if i in s.chain_sent or i not in s.chain_frames or node not in s.swaps:
            return
        s.chain_sent.add(i)
        _, ok, bits = s.swaps[node]
        if ok:
            self.send_chain(time, s, i, bits ^ s.chain_frames[i])

    def deliver(self, time: float, s: State) -> None:
        if s.ended:
            return
        q = s.plan.route
        consumption = {q[0]: time, q[-1]: time, **{v: t for v, (t, _, _) in s.swaps.items()}}
        exposure = 0.0
        for edge, birth in s.births.items():
            link = self.topo.q[edge]
            exposure += sum(consumption[node] - birth for node in edge) / link.lifetime
        scaled_exposure = exposure / self.condition.memory_scale
        quality = fidelity_from_exposure(exp(s.plan.log_raw), scaled_exposure, self.condition.dephasing)
        self.rows[s.request.number]["storage_exposure"] = scaled_exposure
        reason = "qualified" if quality >= s.request.min_fidelity else "fidelity"
        self.finish(time, s, reason, quality)

    def expire(self, time: float, s: State) -> None:
        if s.ended:
            return
        if len(s.active) != len(s.plan.route):
            reason = "activation"
        elif s.bad_generation:
            reason = "generation"
        elif any(not ok for _, ok, _ in s.swaps.values()):
            reason = "swap"
        else:
            reason = "deadline_or_control_loss"
        self.finish(s.request.cutoff, s, reason)

    def release(self, time: float, s: State, node: int) -> None:
        count = s.held.pop(node)
        self.free[node] += count
        s.memory_time += count * (time - s.book)
        assert self.free[node] <= self.topo.slots[node]

    def finish(self, time: float, s: State, status: str, quality: float | None = None) -> None:
        s.ended = True
        for node in tuple(s.held):
            self.release(time, s, node)
        q_ready = max((v[0] for v in s.swaps.values()), default=max(s.births.values(), default=None))
        frame = 0
        for bits in s.received.values():
            frame ^= bits
        self.rows[s.request.number].update({
            "status": status, "terminal_ms": time - s.request.offer_ms,
            "fidelity": quality, "memory_slot_ms": s.memory_time, "frame": frame if quality else None,
            "quantum_ready_ms": q_ready - s.request.offer_ms if quality is not None else None,
        })

    def row(self, request: Request, status: str, admitted: bool, queue_ms: float,
            measured: float, predicted: float | None = None) -> dict:
        return {
            **asdict(request), "status": status, "admitted": admitted,
            "controller_wait_ms": queue_ms, "selection_cpu_ms": measured,
            "predicted": predicted, "route": None, "detours": None,
            "selection_applied_ms": None, "dispatch_ms": None,
            "terminal_ms": None, "fidelity": None, "memory_slot_ms": 0.0,
            "frame": None, "quantum_ready_ms": None, "storage_exposure": None,
            "reserved_byte_hops": None,
        }

    def run(self) -> tuple[list[dict], dict]:
        self.net.run()
        assert self.free == self.topo.slots
        result = [self.rows[request.number] for request in self.requests]
        assert all(row["status"] != "pending" for row in result)
        assert all(row["terminal_ms"] <= row["deadline_ms"] + 1e-8
                   for row in result if row["terminal_ms"] is not None)
        counters = {
            "bytes_by_type": dict(self.net.bytes),
            "drops_by_type": dict(self.net.drops),
            "background_offered": self.net.bg_ingress,
            "background_delivered": self.net.bg_delivered,
            "peak_used_slots": self.peak_used_slots,
        }
        return result, counters
