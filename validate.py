"""Cross-check simulator physics, FIFO service, and factored route scores."""

from __future__ import annotations

import argparse
import json
from collections import deque
from itertools import pairwise
from pathlib import Path
from random import Random

import numpy as np

from campaign import choose_pairs
from network import BUFFER, enqueue, make_topology
from planner import Catalog, FactoredScores
from simulator import fidelity_from_exposure

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
PHI = np.array([1, 0, 0, 1]) / np.sqrt(2)
BELL = np.outer(PHI, PHI)


def channel(state, half, retention, dephasing):
    if dephasing:
        operators = (I, Z)
        weights = ((1 + retention) / 2, (1 - retention) / 2)
    else:
        operators = (I, X, Y, Z)
        weights = ((1 + 3 * retention) / 4,) + ((1 - retention) / 4,) * 3
    result = np.zeros((4, 4), dtype=complex)
    for operator, weight in zip(operators, weights, strict=True):
        local = np.kron(operator, I) if half == 0 else np.kron(I, operator)
        result += weight * (local @ state @ local.conj().T)
    return result


def swap(left, right):
    joint = np.kron(left, right).reshape((2,) * 8)
    bell = PHI.reshape(2, 2)
    projected = np.einsum("bc,abcdefgh,fg->adeh", bell, joint, bell).reshape(4, 4)
    return projected / np.trace(projected)


def density_check():
    rng = np.random.default_rng(691250)
    error = 0.0
    cases = 0
    for dephasing in (False, True):
        for link_count in range(1, 6):
            for _ in range(32):
                visibility = rng.uniform(0.8, 1, link_count)
                exposure = rng.uniform(0, 0.3, (link_count, 2))
                swap_visibility = rng.uniform(0.93, 1, link_count - 1)
                state = None
                for i, v in enumerate(visibility):
                    link = v * BELL + (1 - v) * np.eye(4) / 4
                    for half, age in enumerate(exposure[i]):
                        link = channel(link, half, np.exp(-age), dephasing)
                    if state is None:
                        state = link
                    else:
                        state = channel(swap(state, link), 0, swap_visibility[i - 1], False)
                measured = float((PHI @ state @ PHI).real)
                predicted = fidelity_from_exposure(
                    float(visibility.prod() * swap_visibility.prod()),
                    float(exposure.sum()), dephasing,
                )
                error = max(error, abs(measured - predicted))
                cases += 1
    assert error < 1e-12
    return {"cases": cases, "max_absolute_error": error}


def fifo_check():
    rng = Random(512911)
    cases = drops = 0
    worst = 0.0
    for rate in (0.30, 0.75, 1.10, 2.20):
        queue = deque()
        finishes = []
        last_finish = now = 0.0
        for _ in range(2_500):
            now += rng.expovariate(rate)
            service = rng.uniform(0.04, 2.0)
            live = sum(t > now for t in finishes)
            reference = None
            if live < BUFFER:
                last_finish = max(now, last_finish) + service
                reference = last_finish
                finishes.append(last_finish)
            value = enqueue(queue, now, service)
            assert (reference is None) == (value is None)
            if reference is not None:
                worst = max(worst, abs(reference - value))
            else:
                drops += 1
            cases += 1
    assert drops > 0 and worst < 1e-12
    return {"cases": cases, "drops": drops, "max_finish_error_ms": worst}


def allowance_check():
    rng = np.random.default_rng(427711)
    v0 = rng.uniform(.5, 1, 20000)
    exposure = rng.uniform(0, .4, len(v0))
    decay = rng.uniform(.01, .12, len(v0))
    quantum_ready = rng.uniform(0, 10, len(v0))
    deadline = rng.uniform(2, 17, len(v0))
    minimum = rng.uniform(.58, .96, len(v0))
    arrivals = rng.uniform(0, 14, (len(v0), 3))
    wait = np.maximum(0, arrivals.max(axis=1) - quantum_ready)
    limit = np.minimum(deadline - quantum_ready,
                       (np.log(v0 / ((4 * minimum - 1) / 3)) - exposure) / decay)
    direct = ((quantum_ready + wait <= deadline)
              & ((1 + 3 * v0 * np.exp(-exposure - decay * wait)) / 4 >= minimum))
    assert np.array_equal(wait <= limit, direct)
    return {"cases": len(v0), "qualified": int(direct.sum())}


def factor_check():
    rng = np.random.default_rng(127817)
    comparisons = 0
    for topo_name in ("synthetic-24-1", "synthetic-36-2", "abilene", "polska"):
        topo = make_topology(topo_name)
        catalog = Catalog(topo, .73)
        pairs = choose_pairs(topo)
        catalog.warm(pairs)
        for pair in pairs:
            for _ in range(20):
                target = np.log((4 * rng.uniform(.82, .94) - 1) / 3)
                remaining = rng.uniform(6, 30)
                scale = rng.uniform(.5, 1.2)
                shifts = {edge: rng.uniform(0, 2) for edge in topo.c}

                def shift(route):
                    return sum(shifts[edge] for edge in pairwise(route))

                scorer = FactoredScores(target, remaining, scale, catalog.n)
                for plan in catalog.table[pair]:
                    for method, padding in (("joint", 0), ("guarded", .65)):
                        if plan.controls:
                            delay = [shift(route) + padding * (len(route) - 1)
                                     for route in plan.controls]
                            latest = np.maximum(plan.ready, (plan.arrivals + np.array(delay)[:, None]).max(axis=0))
                        else:
                            latest = plan.ready
                        loss = (plan.pre_exposure + plan.endpoint_rate * (latest - plan.ready)) / scale
                        direct = np.mean(plan.valid & (latest <= remaining) & (plan.log_raw - loss >= target))
                        assert scorer.score(plan, method, shift) == direct
                        comparisons += 1
    return {"cases": comparisons, "max_absolute_score_error": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/validation.json"))
    args = parser.parse_args()
    output = {
        "density_matrix": density_check(), "fifo": fifo_check(),
        "allowance": allowance_check(), "factored_scores": factor_check(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
