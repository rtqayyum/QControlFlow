"""Topology, independent traffic tapes, and finite FIFO packet service."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from hashlib import blake2b
from heapq import heappop, heappush
from itertools import count, pairwise
import json
from math import exp, hypot
from pathlib import Path
from random import Random
from typing import Callable

Edge = tuple[int, int]
PathNodes = tuple[int, ...]
ACT_BYTES = 80
TRACK_BYTES = 96
BG_BYTES = 1500
BUFFER = 16
ATTEMPTS = 4
SWAP_MS = 0.18
SWAP_SUCCESS = 0.995
SWAP_VISIBILITY = 0.998


def seed_for(*parts: object) -> int:
    key = ":".join(map(str, parts)).encode()
    return int.from_bytes(blake2b(key, digest_size=8).digest(), "big")


def edge_key(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def visibility(fidelity: float) -> float:
    return (4 * fidelity - 1) / 3


@dataclass(frozen=True)
class QuantumLink:
    ms: float
    p: float
    fidelity: float
    lifetime: float


@dataclass(frozen=True)
class ClassicalLink:
    mbps: float
    propagation: float
    offered_factor: float
    risk: int

    def service(self, size: int) -> float:
        return 8 * size / (1000 * self.mbps)


@dataclass
class Topology:
    name: str
    q: dict[Edge, QuantumLink]
    c: dict[Edge, ClassicalLink]
    q_adj: dict[int, tuple[int, ...]]
    c_adj: dict[int, tuple[int, ...]]
    slots: dict[int, int]
    hub: int = 0


def adjacency(n: int, edges: set[Edge]) -> dict[int, tuple[int, ...]]:
    return {v: tuple(sorted(b for a, b in edges if a == v)) for v in range(n)}


def make_topology(name: str) -> Topology:
    rng = Random(seed_for("topology", name))
    if name.startswith("synthetic-"):
        n = int(name.split("-")[1])
        coords = {v: (rng.random(), rng.random()) for v in range(n)}
        q_edges = {edge_key(v, (v + 1) % n) for v in range(n)}
        candidates = [(a, b) for a in range(n) for b in range(a + 1, n)
                      if (a, b) not in q_edges]
        rng.shuffle(candidates)
        q_edges.update(candidates[:n // 2])
    else:
        raw = json.loads((Path(__file__).parent / "data" / f"{name}.json").read_text())
        n = len(raw["nodes"])
        q_edges = {edge_key(a, b) for a, b in raw["edges"]}
        coords = {v: (rng.random(), rng.random()) for v in range(n)}

    def length(a: int, b: int) -> float:
        ax, ay = coords[a]
        bx, by = coords[b]
        return 10 + 38 * hypot(ax - bx, ay - by)

    q = {}
    for a, b in sorted(q_edges):
        km = length(a, b)
        q[a, b] = QuantumLink(
            0.60 + 0.022 * km,
            min(0.94, (0.88 + 0.18 * rng.random()) * exp(-km / 180)),
            0.983 + 0.014 * rng.random(),
            rng.uniform(65, 220),
        )
    extra = [(a, b) for a in range(n) for b in range(a + 1, n)
             if (a, b) not in q_edges]
    rng.shuffle(extra)
    c_undirected = q_edges | set(extra[:n // 3])
    c = {}
    for a, b in sorted(c_undirected):
        # These are experimental parameters, including for named adjacencies.
        cap = rng.choice((8.0, 12.0, 20.0, 32.0))
        risk = (a + b) % 5
        for u, v in ((a, b), (b, a)):
            c[u, v] = ClassicalLink(
                cap, 0.07 + 0.0049 * length(a, b), rng.uniform(0.58, 1.12), risk,
            )
    q_directed = {(u, v) for a, b in q for u, v in ((a, b), (b, a))}
    slots = {v: rng.choice((4, 6, 8)) for v in range(n)}
    return Topology(name, q, c, adjacency(n, q_directed), adjacency(n, set(c)), slots)


def paths(
    adj: dict[int, tuple[int, ...]], source: int, dest: int,
    cost: Callable[[int, int], float], limit: int, max_hops: int,
    forbidden: Edge | None = None,
) -> list[PathNodes]:
    if source == dest:
        return [(source,)]
    queue = [(0.0, (source,))]
    found = []
    while queue and len(found) < limit:
        weight, route = heappop(queue)
        node = route[-1]
        if node == dest:
            found.append(route)
            continue
        if len(route) > max_hops:
            continue
        for nxt in adj[node]:
            if nxt not in route and (node, nxt) != forbidden:
                heappush(queue, (weight + cost(node, nxt), route + (nxt,)))
    return found


def burst_arrivals(rng: Random, link: ClassicalLink, load: float,
                   start: float, stop: float) -> list[float]:
    phase = rng.uniform(0.0, 5.0)
    now = start
    end = now + phase
    rate = min(1.12, load * link.offered_factor) / link.service(BG_BYTES)
    heavy = rng.randrange(2)
    arrivals = []
    while now < stop:
        boundary = min(end, stop)
        burst_rate = rate * (1.55 if heavy else 0.45)
        if burst_rate > 0:
            time = now + rng.expovariate(burst_rate)
            while time < boundary:
                arrivals.append(time)
                time += rng.expovariate(burst_rate)
        now, end = end, end + 5.0
        heavy = 1 - heavy
    return arrivals


def enqueue(queue: deque[float], time: float, service: float) -> float | None:
    while queue and queue[0] <= time:
        queue.popleft()
    if len(queue) >= BUFFER:
        return None
    finish = max(time, queue[-1]) + service if queue else time + service
    queue.append(finish)
    return finish


def residual(queue: deque[float], time: float) -> float:
    return max(0.0, queue[-1] - time) if queue else 0.0


def calibration_waits(topology: Topology, load: float, samples: int) -> dict[Edge, list[float]]:
    waits = {}
    for edge, link in topology.c.items():
        rng = Random(seed_for(topology.name, load, edge, "calibration"))
        arrivals = burst_arrivals(rng, link, load, -40, 620)
        probes = sorted(rng.uniform(0, 620) for _ in range(samples))
        queue: deque[float] = deque()
        i = 0
        values = []
        for time in probes:
            while i < len(arrivals) and arrivals[i] < time:
                enqueue(queue, arrivals[i], link.service(BG_BYTES))
                i += 1
            values.append(residual(queue, time))
        rng.shuffle(values)
        waits[edge] = values
    return waits


@dataclass
class Packet:
    route: PathNodes
    size: int
    kind: str
    finish: Callable[[float], None] | None = None


class Network:
    def __init__(self, topo: Topology, outage: tuple[int, float, float] | None) -> None:
        self.topo = topo
        self.outage = outage
        self.heap = []
        self.order = count()
        self.queues = {edge: deque() for edge in topo.c}
        self.bytes: Counter[str] = Counter()
        self.drops: Counter[str] = Counter()
        self.bg_ingress = 0
        self.bg_delivered = 0

    def at(self, time: float, task: Callable, *args: object) -> None:
        heappush(self.heap, (time, next(self.order), task, args))

    def run(self) -> None:
        while self.heap:
            time, _, task, args = heappop(self.heap)
            task(time, *args)

    def send(self, time: float, packet: Packet, hop: int = 0) -> None:
        if hop == len(packet.route) - 1:
            if packet.kind == "background":
                self.bg_delivered += 1
            if packet.finish:
                packet.finish(time)
            return
        edge = (packet.route[hop], packet.route[hop + 1])
        link = self.topo.c[edge]
        finish = enqueue(self.queues[edge], time, link.service(packet.size))
        if finish is None:
            self.drops[packet.kind] += 1
            return
        self.bytes[packet.kind] += packet.size
        if self.outage and packet.kind != "background":
            group, begin, end = self.outage
            if link.risk == group and begin <= finish < end:
                self.drops[packet.kind] += 1
                return
        self.at(finish + link.propagation, self.send, packet, hop + 1)

    def load_background(self, load: float, seed: int, stop: float) -> None:
        for edge, link in self.topo.c.items():
            for time in burst_arrivals(Random(seed_for(seed, edge)), link, load, -18, stop):
                self.bg_ingress += 1
                self.at(time, self.send, Packet(edge, BG_BYTES, "background"))
