"""A common route catalog and offline-calibrated deadline/fidelity forecast."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, pairwise
from math import log
from statistics import median

import numpy as np

from network import (
    ACT_BYTES, ATTEMPTS, SWAP_MS, SWAP_SUCCESS, SWAP_VISIBILITY, TRACK_BYTES,
    Edge, PathNodes, Topology, calibration_waits, edge_key, paths, seed_for, visibility,
)

BYTE_CAP = 3000
SAMPLES = 40
TELEMETRY_MS = 9.0
CHOICES = ("qsp", "sequential", "mean", "single", "static", "joint", "chain",
           "guarded", "replicate", "retry", "gated",
           "fast_sequential", "fast_joint", "fast_gated")


@dataclass
class Forecast:
    route: PathNodes
    controls: tuple[PathNodes, ...]
    backups: tuple[PathNodes, ...]
    activation: tuple[PathNodes, ...]
    chain: tuple[PathNodes, ...]
    detours: int
    risk_bytes: int
    qubits: dict[int, int]
    ready: np.ndarray
    pre_exposure: np.ndarray
    endpoint_rate: float
    log_raw: float
    valid: np.ndarray
    arrivals: np.ndarray


@dataclass(frozen=True)
class Decision:
    plan: Forecast | None
    predicted: float | None
    reason: str | None


class FactoredScores:
    """Cache the scenario pass sets shared by plans with common routes."""

    def __init__(self, target: float, remaining: float, scale: float, samples: int) -> None:
        self.target = target
        self.remaining = remaining
        self.scale = scale
        self.samples = samples
        self.cutoffs: dict[PathNodes, tuple[np.ndarray, int]] = {}
        self.masks: dict[tuple[PathNodes, int, PathNodes, str], int] = {}

    @staticmethod
    def mask(values: np.ndarray) -> int:
        # Each Boolean occupies a byte containing zero or one. Bit count
        # therefore counts scenarios without an extra bit-packing allocation.
        return int.from_bytes(values.tobytes(), byteorder="little")

    def score(self, plan: Forecast, method: str, shift) -> float:
        if plan.route not in self.cutoffs:
            spare = (plan.log_raw - self.target) * self.scale - plan.pre_exposure
            latest = np.minimum(self.remaining, plan.ready + spare / plan.endpoint_rate)
            feasible = plan.valid & (plan.ready <= self.remaining) & (spare >= 0)
            self.cutoffs[plan.route] = latest, self.mask(feasible)
        latest, common = self.cutoffs[plan.route]
        for index, route in enumerate(plan.controls):
            if not common:
                break
            key = (plan.route, index, route, method)
            if key not in self.masks:
                delay = shift(route)
                if method == "guarded":
                    delay += 0.65 * (len(route) - 1)
                self.masks[key] = self.mask(plan.arrivals[index] + delay <= latest)
            common &= self.masks[key]
        return common.bit_count() / self.samples


class Catalog:
    def __init__(self, topo: Topology, load: float, samples: int = SAMPLES) -> None:
        self.topo = topo
        self.load = load
        self.n = samples
        self.waits = {edge: np.array(values)
                      for edge, values in calibration_waits(topo, load, samples).items()}
        self.med = {edge: median(values) for edge, values in self.waits.items()}
        self.path_cache: dict[tuple[int, int], tuple[PathNodes, PathNodes]] = {}
        self.table: dict[tuple[int, int], tuple[Forecast, ...]] = {}

    def cpaths(self, a: int, b: int) -> tuple[PathNodes, PathNodes]:
        key = (a, b)
        if key not in self.path_cache:
            def cost(u: int, v: int) -> float:
                link = self.topo.c[u, v]
                return link.propagation + link.service(TRACK_BYTES)
            routes = paths(self.topo.c_adj, a, b, cost, 5, 10)
            default = routes[0]
            edges = set(pairwise(default))
            alternatives = routes[1:]
            alternate = min(alternatives, key=lambda r: (len(set(pairwise(r)) & edges),
                                                          sum(cost(x, y) for x, y in pairwise(r)))) if alternatives else default
            self.path_cache[key] = (default, alternate)
        return self.path_cache[key]

    def delay(self, route: PathNodes, bytes_: int) -> np.ndarray:
        total = np.zeros(self.n)
        for edge in pairwise(route):
            link = self.topo.c[edge]
            total += self.waits[edge] + link.propagation + link.service(bytes_)
        return total

    def compile(self, source: int, dest: int) -> tuple[Forecast, ...]:
        def cost(u: int, v: int) -> float:
            link = self.topo.q[edge_key(u, v)]
            return link.ms / link.p

        qpaths = paths(self.topo.q_adj, source, dest, cost, 3, 6)
        forecasts = []
        for qpath in qpaths:
            repeaters = qpath[1:-1]
            activation = tuple(self.cpaths(self.topo.hub, node)[0] for node in qpath)
            setup = {node: self.delay(route, ACT_BYTES) for node, route in zip(qpath, activation)}
            births = {}
            valid = np.ones(self.n, dtype=bool)
            raw = len(repeaters) * log(SWAP_VISIBILITY)
            qubits: Counter[int] = Counter()
            edges = [edge_key(a, b) for a, b in pairwise(qpath)]
            for a, b in edges:
                link = self.topo.q[a, b]
                rng = np.random.default_rng(seed_for("forecast", self.topo.name, source, dest, a, b))
                tries = rng.geometric(link.p, self.n)
                valid &= tries <= ATTEMPTS
                births[a, b] = np.maximum(setup[a], setup[b]) + np.minimum(tries, ATTEMPTS) * link.ms
                raw += log(visibility(link.fidelity))
                qubits.update((a, b))
            swaps = {}
            for i, node in enumerate(repeaters):
                swaps[node] = np.maximum(births[edges[i]], births[edges[i + 1]]) + SWAP_MS
                rng = np.random.default_rng(seed_for("forecast", self.topo.name, source, dest, node, "swap"))
                valid &= rng.random(self.n) < SWAP_SUCCESS
            ready = np.maximum.reduce(list(swaps.values()) if swaps else list(births.values()))
            release = {source: ready, dest: ready, **swaps}
            exposure = np.zeros(self.n)
            for a, b in edges:
                exposure += (release[a] + release[b] - 2 * births[a, b]) / self.topo.q[a, b].lifetime
            rate = 1 / self.topo.q[edges[0]].lifetime + 1 / self.topo.q[edges[-1]].lifetime
            options = [self.cpaths(node, dest) for node in repeaters]
            eligible = tuple(i for i, (primary, alt) in enumerate(options) if primary != alt)
            bundles = [()] + [(i,) for i in eligible] + list(combinations(eligible, 2))
            if len(eligible) > 2:
                bundles.append(eligible)
            chain = tuple(self.cpaths(a, b)[0] for a, b in pairwise(qpath))
            act_hops = sum(len(route) - 1 for route in activation)
            for bundle in bundles:
                controls = tuple(pair[int(i in bundle)] for i, pair in enumerate(options))
                backups = tuple(pair[int(i not in bundle)] for i, pair in enumerate(options))
                bound = ACT_BYTES * act_hops + TRACK_BYTES * sum(len(p) - 1 for p in controls)
                if bound > BYTE_CAP:
                    continue
                arrivals = np.array([swaps[node] + self.delay(p, TRACK_BYTES)
                                     for node, p in zip(repeaters, controls)])
                forecasts.append(Forecast(qpath, controls, backups, activation, chain,
                                          len(bundle), bound, dict(qubits), ready, exposure,
                                          rate, raw, valid, arrivals))
        return tuple(forecasts)

    def warm(self, pairs: list[tuple[int, int]]) -> None:
        for pair in pairs:
            self.table[pair] = self.compile(*pair)

    def decide(
        self, source: int, dest: int, remaining: float, minimum: float,
        scale: float, slots: dict[int, int], now: float,
        seen: dict[Edge, tuple[float, float]], policy: str, cold: bool = False,
    ) -> Decision:
        factored = policy.startswith("fast_")
        if factored:
            policy = policy.removeprefix("fast_")
        plans = self.compile(source, dest) if cold else self.table[source, dest]
        fit = [p for p in plans if all(slots[node] >= needed for node, needed in p.qubits.items())]
        if not fit:
            return Decision(None, None, "capacity")
        defaults = [p for p in fit if p.detours == 0]
        if policy == "chain":
            defaults = [
                p for p in defaults
                if ACT_BYTES * sum(len(r) - 1 for r in p.activation)
                + TRACK_BYTES * sum(len(r) - 1 for r in p.chain) <= BYTE_CAP
            ]
        if not defaults:
            return Decision(None, None, "budget")

        target = log(visibility(minimum))
        tail: dict[PathNodes, float] = {}

        def shift(route: PathNodes) -> float:
            if route not in tail:
                value = 0.0
                for edge in pairwise(route):
                    if edge in seen:
                        stamp, work = seen[edge]
                        age = now - stamp
                        if age < 2 * TELEMETRY_MS:
                            value += max(0.0, work - age - self.med[edge])
                tail[route] = value
            return tail[route]

        scores: dict[tuple[int, str], float] = {}
        factored_scores = FactoredScores(target, remaining, scale, self.n) if factored else None

        def score(p: Forecast, method: str) -> float:
            key = (id(p), method)
            if key in scores:
                return scores[key]
            if factored_scores is not None:
                result = factored_scores.score(p, method, shift)
                scores[key] = result
                return result
            if not p.controls:
                frame = p.ready
            else:
                delays = np.array([shift(r) if method != "static" else 0.0 for r in p.controls])
                if method == "guarded":
                    delays += np.array([0.65 * (len(r) - 1) for r in p.controls])
                times = p.arrivals + delays[:, None]
                last = times.mean(axis=0) if method == "mean" else times.max(axis=0)
                frame = np.maximum(last, p.ready)
            loss = (p.pre_exposure + p.endpoint_rate * (frame - p.ready)) / scale
            result = float(np.mean(p.valid & (frame <= remaining) & (p.log_raw - loss >= target)))
            scores[key] = result
            return result

        def best(candidates: list[Forecast], method: str) -> tuple[Forecast, float]:
            candidates_scores = [(p, score(p, method)) for p in candidates]
            return min(candidates_scores, key=lambda item: (-item[1], item[0].risk_bytes,
                                                           item[0].route, item[0].controls))

        if policy == "qsp":
            plan = defaults[0]
            return Decision(plan, score(plan, "joint"), None)
        if policy == "gated":
            base, _ = best(defaults, "joint")
            sequential, baseline = best([p for p in fit if p.route == base.route], "joint")
            if baseline >= 0.90:
                return Decision(sequential, baseline, None)
            conservative = score(sequential, "guarded")
            challengers = [
                p for p in fit if p.route != base.route
                and score(p, "joint") - baseline >= 0.10 - 1e-9
                and score(p, "guarded") - conservative >= 0.025 - 1e-9
            ]
            if challengers:
                challenger, probability = best(challengers, "joint")
                return Decision(challenger, probability, None)
            return Decision(sequential, baseline, None)
        if policy in ("sequential", "chain"):
            base, probability = best(defaults, "joint")
            if policy == "chain":
                return Decision(base, probability, None)
            fit = [p for p in fit if p.route == base.route]
        if policy == "single":
            fit = [p for p in fit if p.detours <= 1]
        method = policy if policy in ("mean", "static", "guarded") else "joint"
        chosen, probability = best(fit, method)
        if policy == "guarded" and probability < 0.40:
            return Decision(None, probability, "risk_reject")
        return Decision(chosen, probability, None)
