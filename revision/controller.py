"""Batch forecasts, explicit outcome self-contention, and a compute-aware gate."""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count, pairwise
from math import log

import numpy as np

from network import TRACK_BYTES, Topology, visibility
from planner import Catalog, Forecast, TELEMETRY_MS


def coupled_arrivals(topo, controls, emission, waits):
    """Serialize outcomes against common sampled residual-work floors.

    This is a frozen-background forecast, not an exact live FIFO model.
    Unlike the original marginal forecast, one outcome can delay another.
    No future live network state or outage schedule is consulted.
    """
    n = emission.shape[1]
    arrivals = np.empty_like(emission)
    for scenario in range(n):
        queue_end = {}
        order = count()
        heap = [(float(emission[i, scenario]), next(order), i, 0)
                for i in range(len(controls))]
        from heapq import heapify
        heapify(heap)
        while heap:
            time, _, i, hop = heappop(heap)
            route = controls[i]
            if hop == len(route) - 1:
                arrivals[i, scenario] = time
                continue
            edge = route[hop], route[hop + 1]
            link = topo.c[edge]
            start = max(time + float(waits[edge][scenario]), queue_end.get(edge, -np.inf))
            finish = start + link.service(TRACK_BYTES)
            queue_end[edge] = finish
            heappush(heap, (finish + link.propagation, next(order), i, hop + 1))
    return arrivals


def feature_key(load, scale, outcomes):
    load_group = 0 if load < .5 else 1 if load < .85 else 2
    return f"{load_group}:{int(scale < .8)}:{min(int(outcomes), 4)}"


class Calibrator:
    """Shrunk empirical reliability curves fitted only to training traces."""
    def __init__(self, values=None):
        self.values = values or {"global": [i / 10 for i in range(11)], "cells": {}}

    @staticmethod
    def monotone(values, weights):
        blocks = []
        for index, (value, weight) in enumerate(zip(values, weights)):
            blocks.append([index, index, float(value) * weight, float(weight)])
            while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
                right = blocks.pop()
                left = blocks.pop()
                blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
        result = [0.] * len(values)
        for a, b, total, weight in blocks:
            result[a:b + 1] = [total / weight] * (b - a + 1)
        return result

    @classmethod
    def fit(cls, rows):
        groups = {"global": np.zeros((11, 2))}
        for r in rows:
            if not r["admitted"] or r.get("raw_predicted") is None:
                continue
            bucket = min(10, max(0, int(round(float(r["raw_predicted"]) * 10))))
            cell = feature_key(float(r["nominal_load"]), float(r["memory_scale"]), int(r["outcomes"]))
            for key in ["global", cell]:
                data = groups.setdefault(key, np.zeros((11, 2)))
                data[bucket, 0] += 1
                data[bucket, 1] += r["status"] == "qualified"
        raw = groups.pop("global")
        prior = np.arange(11) / 10
        curve = cls.monotone((raw[:, 1] + 8 * prior) / (raw[:, 0] + 8), raw[:, 0] + 8)
        cells = {}
        for key, data in groups.items():
            estimate = (data[:, 1] + 32 * np.array(curve)) / (data[:, 0] + 32)
            cells[key] = cls.monotone(estimate, data[:, 0] + 32)
        return cls({"global": curve, "cells": cells, "training_admissions": int(raw[:, 0].sum())})

    def predict(self, values, load, scale, outcomes):
        key = feature_key(load, scale, outcomes)
        curve = self.values["cells"].get(key, self.values["global"])
        return np.interp(values, np.arange(11) / 10, curve)


@dataclass
class Selection:
    plan: Forecast | None
    raw: float | None
    predicted: float | None
    mode: str
    cost_ms: float
    reason: str | None = None


class Batch:
    def __init__(self, catalog: Catalog, plans):
        self.catalog = catalog
        self.plans = tuple(plans)
        self.nodes = sorted({v for p in plans for v in p.qubits})
        self.needs = np.array([[p.qubits.get(v, 0) for v in self.nodes] for p in plans])
        self.defaults = np.array([p.detours == 0 for p in plans])
        self.ready = np.array([p.ready for p in plans])
        self.exposure = np.array([p.pre_exposure for p in plans])
        self.rate = np.array([p.endpoint_rate for p in plans])[:, None]
        self.raw = np.array([p.log_raw for p in plans])[:, None]
        self.valid = np.array([p.valid for p in plans])
        self.outcome_counts = np.array([len(p.controls) for p in plans])
        self.k = max((len(p.controls) for p in plans), default=0)
        self.routes = sorted({r for p in plans for r in p.controls})
        route_index = {r: i for i, r in enumerate(self.routes)}
        self.edges = tuple(catalog.topo.c)
        edge_index = {edge: i for i, edge in enumerate(self.edges)}
        self.incidence = np.zeros((len(self.routes), len(self.edges)))
        for i, route in enumerate(self.routes):
            for edge in pairwise(route):
                self.incidence[i, edge_index[edge]] += 1
        self.route_ids = np.full((len(plans), self.k), len(self.routes), dtype=int)
        self.independent = np.full((len(plans), self.k, catalog.n), -np.inf)
        self.coupled = self.independent.copy()
        for index, p in enumerate(plans):
            k = len(p.controls)
            if not k:
                continue
            self.route_ids[index, :k] = [route_index[r] for r in p.controls]
            self.independent[index, :k] = p.arrivals
            emission = np.array([p.arrivals[i] - catalog.delay(r, TRACK_BYTES)
                                 for i, r in enumerate(p.controls)])
            self.coupled[index, :k] = coupled_arrivals(catalog.topo, p.controls, emission, catalog.waits)
        self.tie_order = sorted(range(len(plans)), key=lambda i: (
            plans[i].risk_bytes, plans[i].route, plans[i].controls))

    def feasible(self, free):
        return np.all(self.needs <= np.array([free[v] for v in self.nodes]), axis=1)

    def shifts(self, now, seen):
        residual = np.zeros(len(self.edges))
        for i, edge in enumerate(self.edges):
            if edge in seen:
                stamp, work = seen[edge]
                age = now - stamp
                if age < 2 * TELEMETRY_MS:
                    residual[i] = max(0., work - age - self.catalog.med[edge])
        # Avoid a BLAS thread launch for a tiny matrix-vector product.
        return np.append(np.sum(self.incidence * residual, axis=1), 0.)

    def score(self, remaining, minimum, scale, shifts, indices=None, coupled=True):
        indices = np.arange(len(self.plans)) if indices is None else np.asarray(indices)
        ready = self.ready[indices]
        frame = ready
        if self.k:
            arrivals = (self.coupled if coupled else self.independent)[indices]
            delay = shifts[self.route_ids[indices]][:, :, None]
            frame = np.maximum(ready, np.max(arrivals + delay, axis=1))
        loss = (self.exposure[indices] + self.rate[indices] * (frame - ready)) / scale
        return np.mean(self.valid[indices] & (frame <= remaining)
                       & (self.raw[indices] - loss >= log(visibility(minimum))), axis=1)

    def quantum_upper(self, remaining, minimum, scale):
        return np.mean(self.valid & (self.ready <= remaining)
                       & (self.raw - self.exposure / scale >= log(visibility(minimum))), axis=1)

    def best(self, scores, feasible):
        best = None
        for index in self.tie_order:
            if feasible[index] and (best is None or scores[index] > scores[best]):
                best = index
        return best


class Controller:
    def __init__(self, catalog, costs=None, calibrator=None):
        self.catalog = catalog
        self.batches = {pair: Batch(catalog, plans) for pair, plans in catalog.table.items()}
        self.costs = costs or {"base": .05, "full": .15}
        self.calibrator = calibrator or Calibrator()
        self.curves = {}

    def calibration_curves(self, b, scale):
        key = id(b), int(scale < .8)
        if key not in self.curves:
            self.curves[key] = np.array([self.calibrator.values["cells"].get(
                feature_key(self.catalog.load, scale, k), self.calibrator.values["global"])
                for k in b.outcome_counts])
        return self.curves[key]

    def decide(self, source, dest, remaining, minimum, scale, free, now, seen,
               policy, offer_rate=0., feedback=0.):
        b = self.batches[source, dest]
        fit = b.feasible(free)
        defaults = np.flatnonzero(fit & b.defaults)
        if not len(defaults):
            return Selection(None, None, None, "base", self.costs["base"], "capacity")
        base = int(defaults[0])
        shifts = b.shifts(now, seen)
        curves = self.calibration_curves(b, scale)

        def calibrated_array(values):
            x = np.clip(values, 0, 1) * 10
            left = x.astype(int)
            index = np.arange(len(x))
            a = curves[index, left]
            z = curves[index, np.minimum(left + 1, 10)]
            return np.clip(a + (z - a) * (x - left) + feedback, 0, 1)

        def calibrated(p, i):
            x = min(1., max(0., float(p))) * 10
            left = int(x)
            value = curves[i, left] + (curves[i, min(left + 1, 10)] - curves[i, left]) * (x - left)
            return min(1., max(0., float(value) + feedback))
        base_cost, full_cost = self.costs["base"], self.costs["full"]
        couple = policy != "aware_independent"
        subtract = policy != "aware_nocost"
        raw_base = float(b.score(remaining - (base_cost if subtract else 0), minimum,
                                scale, shifts, [base], coupled=couple)[0])
        p_base = calibrated(raw_base, base)
        if policy.startswith("aware"):
            upper = b.quantum_upper(remaining - (full_cost if subtract else 0), minimum, scale)
            optimistic = float(np.max(calibrated_array(upper)[fit]))
            extra = max(0., full_cost - base_cost)
            penalty = offer_rate * extra * p_base if subtract else 0.
            if optimistic - p_base <= penalty + 1e-12:
                return Selection(b.plans[base], raw_base, p_base, "base", base_cost)
        scores = b.score(remaining - (full_cost if subtract else 0), minimum,
                         scale, shifts, coupled=couple)
        if policy == "batch_joint":
            chosen = b.best(scores, fit)
            return Selection(b.plans[chosen], float(scores[chosen]), float(scores[chosen]), "full", full_cost)
        probabilities = calibrated_array(scores)
        chosen = b.best(probabilities, fit)
        # Full computation has already occurred, so even a fallback pays full cost.
        if probabilities[chosen] - p_base <= penalty:
            chosen = base
        return Selection(b.plans[chosen], float(scores[chosen]), float(probabilities[chosen]), "full", full_cost)
