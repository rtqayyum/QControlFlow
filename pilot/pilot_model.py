"""Five-node event model for the first QControlFlow co-routing experiment."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from functools import partial
from heapq import heappop, heappush
from itertools import count, pairwise, product
from math import exp, inf, log
from random import Random
from typing import Callable, Mapping

Edge = tuple[str, str]
Route = tuple[str, ...]

ACTIVATION_BYTES = 64
OUTCOME_BYTES = 96
BACKGROUND_BYTES = 1500
QUEUE_PACKETS = 12
ATTEMPTS = 4
SWAP_MS = 0.15
SWAP_SUCCESS = 0.99
SWAP_VISIBILITY = 0.995
WARMUP_MS = 32.0
BACKGROUND_END_MS = 35.0


@dataclass(frozen=True)
class QuantumLink:
    attempt_ms: float
    success: float
    fidelity: float
    memory_ms: float


QUANTUM = {
    ("A", "R1"): QuantumLink(1.20, 0.80, 0.995, 60.0),
    ("R1", "R2"): QuantumLink(1.30, 0.75, 0.985, 60.0),
    ("R2", "B"): QuantumLink(1.40, 0.80, 0.990, 60.0),
    ("A", "R3"): QuantumLink(1.80, 0.85, 0.955, 150.0),
    ("R3", "B"): QuantumLink(2.10, 0.85, 0.965, 150.0),
}

# Each directed link has its own FIFO. Capacities are Mbps and propagation is ms.
CLASSICAL: dict[Edge, tuple[float, float]] = {
    ("A", "R1"): (40.0, 0.18),
    ("A", "R2"): (40.0, 0.26),
    ("A", "R3"): (40.0, 0.23),
    ("A", "B"): (40.0, 0.34),
    ("R1", "R2"): (40.0, 0.15),
    ("R1", "R3"): (40.0, 0.65),
    ("R2", "R3"): (40.0, 0.70),
    ("R3", "R2"): (40.0, 0.70),
    ("R2", "B"): (8.0, 0.18),
    ("R3", "B"): (8.0, 0.30),
}

QUANTUM_PATHS: dict[str, Route] = {
    "A": ("A", "R1", "R2", "B"),
    "B": ("A", "R3", "B"),
}

OUTCOME_ROUTES: dict[str, tuple[Route, ...]] = {
    "R1": (("R1", "R2", "B"), ("R1", "R3", "B")),
    "R2": (("R2", "B"), ("R2", "R3", "B")),
    "R3": (("R3", "B"), ("R3", "R2", "B")),
}


@dataclass(frozen=True)
class Plan:
    quantum_path: str
    choices: tuple[int, ...]

    @property
    def key(self) -> str:
        return self.quantum_path + "-" + "".join(map(str, self.choices))

    @property
    def nodes(self) -> Route:
        return QUANTUM_PATHS[self.quantum_path]

    @property
    def outcome_routes(self) -> dict[str, Route]:
        return {
            node: OUTCOME_ROUTES[node][choice]
            for node, choice in zip(self.nodes[1:-1], self.choices, strict=True)
        }

    @property
    def reserved_bytes_hops(self) -> int:
        setup = ACTIVATION_BYTES * (len(self.nodes) - 1)
        outcomes = sum((len(route) - 1) * OUTCOME_BYTES for route in self.outcome_routes.values())
        return setup + outcomes

    @property
    def uses_default_routes(self) -> bool:
        return all(choice == 0 for choice in self.choices)


def plan_catalog() -> tuple[Plan, ...]:
    return tuple(
        Plan(path, choices)
        for path, nodes in QUANTUM_PATHS.items()
        for choices in product(*(range(len(OUTCOME_ROUTES[node])) for node in nodes[1:-1]))
    )


@dataclass(frozen=True)
class Workload:
    seed: int
    generation: dict[Edge, tuple[float, ...]]
    swaps: dict[str, tuple[float, int]]
    background: tuple[tuple[float, Edge], ...]


def background_times(rng: Random, mean_rate: float) -> list[float]:
    """Poisson traffic with alternating four-ms intensity and a random phase."""
    t = -WARMUP_MS
    block_end = t + rng.uniform(0.0, 4.0)
    rate_factors = (0.45, 1.55)
    block = rng.randrange(2)
    times: list[float] = []
    while t < BACKGROUND_END_MS:
        end = min(block_end, BACKGROUND_END_MS)
        rate = mean_rate * rate_factors[block]
        if rate > 0:
            arrival = t + rng.expovariate(rate)
            while arrival < end:
                times.append(arrival)
                arrival += rng.expovariate(rate)
        t = block_end
        block_end += 4.0
        block = 1 - block
    return times


def make_workload(seed: int, load: float) -> Workload:
    rng = Random(seed)
    generation = {edge: tuple(rng.random() for _ in range(ATTEMPTS)) for edge in QUANTUM}
    swaps = {node: (rng.random(), rng.randrange(4)) for node in OUTCOME_ROUTES}
    background = []
    for edge, utilization in ((("R2", "B"), load), (("R3", "B"), 0.35 * load)):
        link_rng = Random(rng.getrandbits(64))
        service_ms = 8 * BACKGROUND_BYTES / (1000 * CLASSICAL[edge][0])
        background.extend((time, edge) for time in background_times(link_rng, utilization / service_ms))
    return Workload(seed, generation, swaps, tuple(sorted(background)))


def visibility(fidelity: float) -> float:
    return (4.0 * fidelity - 1.0) / 3.0


def fidelity(werner_visibility: float) -> float:
    return (1.0 + 3.0 * werner_visibility) / 4.0


@dataclass(frozen=True)
class Packet:
    name: str
    route: Route
    size: int
    arrival: Callable[[float], None] | None = None


class PacketNetwork:
    def __init__(self, trace: list[dict] | None = None) -> None:
        self.pending = []
        self.order = count()
        self.finish = {edge: deque() for edge in CLASSICAL}
        self.control_bytes_hops = 0
        self.control_drops = 0
        self.background_arrivals: Counter[Edge] = Counter()
        self.background_drops: Counter[Edge] = Counter()
        self.trace = trace

    def at(self, time: float, action: Callable, *args: object) -> None:
        heappush(self.pending, (time, next(self.order), action, args))

    def log(self, time: float, event: str, **fields: object) -> None:
        if self.trace is not None:
            self.trace.append({"time_ms": time, "event": event, **fields})

    def run(self) -> None:
        while self.pending:
            time, _, action, args = heappop(self.pending)
            action(time, *args)

    def send(self, time: float, packet: Packet, hop: int = 0) -> None:
        if hop == len(packet.route) - 1:
            if packet.arrival is not None:
                self.log(time, "packet_arrival", packet=packet.name, node=packet.route[-1])
                packet.arrival(time)
            return

        edge = (packet.route[hop], packet.route[hop + 1])
        is_background = packet.arrival is None
        if is_background:
            self.background_arrivals[edge] += 1
        queue = self.finish[edge]
        while queue and queue[0] <= time:
            queue.popleft()
        if len(queue) >= QUEUE_PACKETS:
            if is_background:
                self.background_drops[edge] += 1
            else:
                self.control_drops += 1
                self.log(time, "packet_drop", packet=packet.name, edge=edge)
            return

        rate, propagation = CLASSICAL[edge]
        start = queue[-1] if queue else time
        finish = start + 8 * packet.size / (1000 * rate)
        queue.append(finish)
        if not is_background:
            self.control_bytes_hops += packet.size
            self.log(time, "packet_enqueue", packet=packet.name, edge=edge, wait_ms=start - time)
        self.at(finish + propagation, self.send, packet, hop + 1)


@dataclass(frozen=True)
class Outcome:
    failure: str | None
    ready_ms: float | None
    frame_ready_ms: float | None
    raw_visibility: float
    pre_gate_exposure: float
    endpoint_rate: float
    frame: int | None
    control_bytes_hops: int
    control_drops: int

    def delivered_fidelity(self, memory_scale: float) -> float | None:
        if self.frame_ready_ms is None or self.ready_ms is None:
            return None
        exposure = self.pre_gate_exposure + self.endpoint_rate * (self.frame_ready_ms - self.ready_ms)
        return fidelity(self.raw_visibility * exp(-exposure / memory_scale))

    def wait_allowance_ms(self, minimum_fidelity: float, memory_scale: float) -> float | None:
        if self.ready_ms is None:
            return None
        target = visibility(minimum_fidelity)
        if target <= 0:
            return inf
        budget = log(self.raw_visibility / target) - self.pre_gate_exposure / memory_scale
        return budget / (self.endpoint_rate / memory_scale)

    def verdict(self, deadline_ms: float, minimum_fidelity: float, memory_scale: float) -> str:
        if self.failure is not None:
            return self.failure
        if self.frame_ready_ms > deadline_ms:
            return "deadline"
        allowance = self.wait_allowance_ms(minimum_fidelity, memory_scale)
        if self.frame_ready_ms - self.ready_ms > allowance + 1e-10:
            return "fidelity"
        return "qualified"


def simulate(
    plan: Plan,
    tape: Workload,
    planner_ms: float,
    trace: list[dict] | None = None,
    *,
    quantum_links: Mapping[Edge, QuantumLink] = QUANTUM,
) -> Outcome:
    """Execute one request. B collects every local BSM outcome for the pair."""
    network = PacketNetwork(trace)
    nodes = plan.nodes
    edges = tuple(pairwise(nodes))
    adjacent = {node: (edges[i], edges[i + 1]) for i, node in enumerate(nodes[1:-1])}
    outcome_routes = plan.outcome_routes
    active: set[str] = set()
    generating: set[Edge] = set()
    link_ready: dict[Edge, float] = {}
    swapping: set[str] = set()
    swap_completed: dict[str, tuple[float, bool]] = {}
    received: dict[str, tuple[float, bool, int]] = {}

    def receive_outcome(time: float, node: str, success: bool, frame: int) -> None:
        received[node] = (time, success, frame)
        if len(received) == len(adjacent) and all(item[1] for item in received.values()):
            network.log(time, "endpoint_frame_ready", node="B")

    def complete_swap(time: float, node: str) -> None:
        draw, frame = tape.swaps[node]
        success = draw < SWAP_SUCCESS
        swap_completed[node] = (time, success)
        network.log(time, "swap_complete", node=node, success=success, frame=frame)
        callback = lambda arrival: receive_outcome(arrival, node, success, frame)
        network.send(time, Packet(f"outcome:{node}", outcome_routes[node], OUTCOME_BYTES, callback))

    def herald(time: float, edge: Edge, attempt: int) -> None:
        link_ready[edge] = time
        network.log(time, "elementary_pair_ready", edge=edge, attempt=attempt)
        for node in edge:
            if node in adjacent and node not in swapping:
                if all(link in link_ready for link in adjacent[node]):
                    swapping.add(node)
                    network.log(time, "swap_start", node=node)
                    network.at(time + SWAP_MS, complete_swap, node)

    def exhausted(time: float, edge: Edge) -> None:
        network.log(time, "generation_exhausted", edge=edge)

    def activate(node: str, time: float) -> None:
        active.add(node)
        network.log(time, "node_activated", node=node)
        for edge in edges:
            if edge in generating or not all(endpoint in active for endpoint in edge):
                continue
            generating.add(edge)
            link = quantum_links[edge]
            attempt = next(
                (i for i, draw in enumerate(tape.generation[edge], 1) if draw < link.success),
                None,
            )
            network.log(time, "generation_start", edge=edge)
            if attempt is None:
                network.at(time + ATTEMPTS * link.attempt_ms, exhausted, edge)
            else:
                network.at(time + attempt * link.attempt_ms, herald, edge, attempt)

    for time, edge in tape.background:
        network.at(time, network.send, Packet("background", edge, BACKGROUND_BYTES))

    network.at(planner_ms, lambda time: activate("A", time))
    for node in nodes[1:]:
        packet = Packet(f"activate:{node}", ("A", node), ACTIVATION_BYTES, partial(activate, node))
        network.at(planner_ms, network.send, packet)
    network.run()

    def failed(reason: str) -> Outcome:
        return Outcome(reason, None, None, 0.0, 0.0, 0.0, None,
                       network.control_bytes_hops, network.control_drops)

    if len(active) != len(nodes):
        return failed("activation_loss")
    if len(link_ready) != len(edges):
        return failed("generation")
    if not all(swap_completed[node][1] for node in adjacent):
        return failed("swap")

    ready = max(time for time, _ in swap_completed.values())
    release = {node: time for node, (time, _) in swap_completed.items()}
    release.update(A=ready, B=ready)
    raw = SWAP_VISIBILITY ** len(adjacent)
    exposure = 0.0
    for edge in edges:
        link = quantum_links[edge]
        raw *= visibility(link.fidelity)
        exposure += sum(release[node] - link_ready[edge] for node in edge) / link.memory_ms
    endpoint_rate = 1 / quantum_links[edges[0]].memory_ms + 1 / quantum_links[edges[-1]].memory_ms

    if len(received) != len(adjacent):
        return Outcome("outcome_loss", ready, None, raw, exposure, endpoint_rate, None,
                       network.control_bytes_hops, network.control_drops)
    frame_ready = max(item[0] for item in received.values())
    frame = 0
    for _, _, bits in received.values():
        frame ^= bits
    return Outcome(None, ready, frame_ready, raw, exposure, endpoint_rate, frame,
                   network.control_bytes_hops, network.control_drops)
