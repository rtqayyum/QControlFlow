"""Calibrate routing policies, then evaluate on independent matched traffic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from itertools import pairwise, product
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from pilot_model import (
    ACTIVATION_BYTES,
    ATTEMPTS,
    BACKGROUND_BYTES,
    BACKGROUND_END_MS,
    CLASSICAL,
    OUTCOME_BYTES,
    QUANTUM,
    QUANTUM_PATHS,
    QUEUE_PACKETS,
    SWAP_MS,
    SWAP_SUCCESS,
    SWAP_VISIBILITY,
    WARMUP_MS,
    Edge,
    Outcome,
    QuantumLink,
    Plan,
    Workload,
    make_workload,
    plan_catalog,
    simulate,
)

LOADS = (0.25, 0.60, 0.90)
MEMORY_SCALES = (1.0, 2.0)
FIDELITIES = (0.85, 0.90)
DEADLINE_MS = 10.0
BUDGETS = (480, 576)
EXTRA_PLANNER_MS = (0.0, 0.5, 1.0, 2.0, 4.0)
POLICIES = ("default_forwarding", "sequential", "mean_joint", "joint")
REASONS = ("qualified", "deadline", "fidelity", "activation_loss", "generation", "swap", "outcome_loss")


@dataclass(frozen=True)
class Condition:
    load: float
    memory_scale: float
    min_fidelity: float

    def fields(self) -> dict:
        return {**asdict(self), "deadline_ms": DEADLINE_MS}

    def verdict(self, outcome: Outcome) -> str:
        return outcome.verdict(DEADLINE_MS, self.min_fidelity, self.memory_scale)


@dataclass(frozen=True)
class Metrics:
    count: int
    reasons: dict[str, int]
    mean_frame_ms: float | None
    mean_frame_fidelity: float | None
    p95_qualified_ms: float | None
    mean_control_bytes_hops: float
    control_drops: int

    @property
    def qualified(self) -> float:
        return self.reasons["qualified"] / self.count

    def mean_screen_score(self, condition: Condition) -> float:
        if self.mean_frame_ms is None:
            return 0.0
        if self.mean_frame_ms > DEADLINE_MS or self.mean_frame_fidelity < condition.min_fidelity:
            return 0.0
        physical_failures = sum(self.reasons[k] for k in REASONS[3:])
        return (self.count - physical_failures) / self.count


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(index)
    fraction = index - low
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def measure(condition: Condition, outcomes: list[Outcome]) -> Metrics:
    verdicts = [condition.verdict(x) for x in outcomes]
    reasons = Counter(verdicts)
    completed = [x for x in outcomes if x.frame_ready_ms is not None]
    qualified_times = [x.frame_ready_ms for x, v in zip(outcomes, verdicts) if v == "qualified"]
    return Metrics(
        len(outcomes),
        {key: reasons[key] for key in REASONS},
        mean_or_none([x.frame_ready_ms for x in completed]),
        mean_or_none([x.delivered_fidelity(condition.memory_scale) for x in completed]),
        percentile(qualified_times, 0.95) if qualified_times else None,
        mean(x.control_bytes_hops for x in outcomes),
        sum(x.control_drops for x in outcomes),
    )


def choose(
    plans: tuple[Plan, ...], metrics: dict[str, Metrics], condition: Condition, budget: int,
) -> dict[str, Plan]:
    eligible = [plan for plan in plans if plan.reserved_bytes_hops <= budget]

    def best(candidates: list[Plan], objective) -> Plan:
        return min(
            candidates,
            key=lambda plan: (-objective(metrics[plan.key]), plan.reserved_bytes_hops, plan.key),
        )

    def qualified(metric: Metrics) -> float:
        return metric.qualified

    fixed = best([p for p in eligible if p.uses_default_routes], qualified)
    return {
        "default_forwarding": fixed,
        "sequential": best([p for p in eligible if p.quantum_path == fixed.quantum_path], qualified),
        "mean_joint": best(eligible, lambda m: m.mean_screen_score(condition)),
        "joint": best(eligible, qualified),
    }


def seed_lists(seed: int, calibration: int, holdout: int) -> tuple[list[int], list[int]]:
    master = Random(seed)
    calibration_rng, holdout_rng = (Random(master.getrandbits(64)) for _ in range(2))
    cal = [calibration_rng.getrandbits(64) for _ in range(calibration)]
    test = [holdout_rng.getrandbits(64) for _ in range(holdout)]
    if len(set(cal + test)) != calibration + holdout:
        raise ValueError("Seed collision. Use a different master seed.")
    return cal, test


def run_plans(
    plans: tuple[Plan, ...], tapes: list[Workload], planner_ms: float,
    quantum_links: dict[Edge, QuantumLink],
) -> dict[str, list[Outcome]]:
    return {
        plan.key: [simulate(plan, tape, planner_ms, quantum_links=quantum_links) for tape in tapes]
        for plan in plans
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metric_fields(metrics: Metrics) -> dict:
    return {
        "requests": metrics.count,
        "qualified_fraction": metrics.qualified,
        "mean_frame_ms": metrics.mean_frame_ms,
        "mean_frame_fidelity": metrics.mean_frame_fidelity,
        "p95_qualified_ms": metrics.p95_qualified_ms,
        "mean_control_bytes_hops": metrics.mean_control_bytes_hops,
        "control_drops": metrics.control_drops,
        **{f"count_{k}": v for k, v in metrics.reasons.items()},
    }


def boot_interval(differences: list[float], bootstrap: int) -> tuple[float, float]:
    # A seed is the sampling unit because conditions reuse the same exogenous draws.
    rng = Random(12011)
    means = [
        100 * mean(rng.choices(differences, k=len(differences)))
        for _ in range(bootstrap)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def paired_aggregate(
    conditions: list[Condition],
    choices: dict[tuple[Condition, int], dict[str, Plan]],
    test_runs: dict[float, dict[str, list[Outcome]]],
    budget: int,
    bootstrap: int,
) -> dict:
    first_condition = conditions[0]
    first_plan = choices[first_condition, budget]["joint"]
    holdout_count = len(test_runs[first_condition.load][first_plan.key])
    votes = {name: [0] * holdout_count for name in POLICIES}
    wire_bytes = Counter()
    route_changes = 0
    changed_plans = 0
    for condition in conditions:
        policy_plans = choices[condition, budget]
        joint = policy_plans["joint"]
        sequential = policy_plans["sequential"]
        route_changes += joint.quantum_path != sequential.quantum_path
        changed_plans += joint != sequential
        for policy, plan in policy_plans.items():
            for i, outcome in enumerate(test_runs[condition.load][plan.key]):
                votes[policy][i] += condition.verdict(outcome) == "qualified"
                wire_bytes[policy] += outcome.control_bytes_hops
    cells = len(conditions)
    n = cells * len(votes["joint"])
    differences = {}
    for policy in POLICIES[:-1]:
        paired = [(a - b) / cells for a, b in zip(votes["joint"], votes[policy])]
        differences[policy] = {
            "percentage_points": 100 * mean(paired),
            "paired_seed_bootstrap_95_pp": boot_interval(paired, bootstrap),
        }
    return {
        "budget_bytes_hops": budget,
        "offered_requests_per_policy": n,
        "joint_changes_quantum_path_vs_sequential_cells": route_changes,
        "joint_changes_full_plan_vs_sequential_cells": changed_plans,
        "policies": {
            policy: {
                "qualified": sum(votes[policy]),
                "qualified_fraction": sum(votes[policy]) / n,
                "mean_control_bytes_hops": wire_bytes[policy] / n,
            }
            for policy in POLICIES
        },
        "joint_minus": differences,
    }


def planner_sensitivity(
    conditions: list[Condition], choices: dict[tuple[Condition, int], dict[str, Plan]],
    test_runs: dict[float, dict[str, list[Outcome]]], tapes: dict[float, list[Workload]],
    planner_ms: float, quantum_links: dict[Edge, QuantumLink],
) -> list[dict]:
    budget = max(BUDGETS)
    rows = []
    for extra in EXTRA_PLANNER_MS:
        cache = {}
        joint_total, sequential_total, offered = 0, 0, 0
        for condition in conditions:
            policies = choices[condition, budget]
            joint = policies["joint"]
            sequential = policies["sequential"]
            key = (condition.load, joint.key)
            if key not in cache:
                cache[key] = (
                    [simulate(joint, t, planner_ms + extra, quantum_links=quantum_links)
                     for t in tapes[condition.load]]
                    if extra else test_runs[condition.load][joint.key]
                )
            joint_passes = measure(condition, cache[key]).reasons["qualified"]
            sequential_outcomes = test_runs[condition.load][sequential.key]
            sequential_passes = measure(condition, sequential_outcomes).reasons["qualified"]
            n = len(sequential_outcomes)
            joint_total += joint_passes
            sequential_total += sequential_passes
            offered += n
            rows.append({
                "extra_joint_planner_ms": extra,
                "scope": "condition",
                **condition.fields(),
                "offered": n,
                "joint_fraction": joint_passes / n,
                "sequential_fraction": sequential_passes / n,
                "difference_pp": 100 * (joint_passes - sequential_passes) / n,
            })
        rows.append({
            "extra_joint_planner_ms": extra,
            "scope": "all_conditions",
            **{key: "" for key in conditions[0].fields()},
            "offered": offered,
            "joint_fraction": joint_total / offered,
            "sequential_fraction": sequential_total / offered,
            "difference_pp": 100 * (joint_total - sequential_total) / offered,
        })
    return rows


def compare_bundles(
    conditions: list[Condition], test_runs: dict[float, dict[str, list[Outcome]]],
    tapes: dict[float, list[Workload]], planner_ms: float,
    quantum_links: dict[Edge, QuantumLink],
) -> tuple[list[dict], dict]:
    plans = [Plan("A", choices) for choices in ((0, 0), (1, 0), (0, 1), (1, 1))]
    rows = []
    witness = {"description": "No held-out case matched the witness rule."}
    for condition in conditions:
        outcomes = [test_runs[condition.load][p.key] for p in plans]
        successes = [0] * len(plans)
        both_only = 0
        both_rescue = 0
        both_harm = 0
        first_witness = None
        for i, cases in enumerate(zip(*outcomes)):
            verdicts = [condition.verdict(x) for x in cases]
            default, first, second, both = [v == "qualified" for v in verdicts]
            for j, verdict in enumerate(verdicts):
                successes[j] += verdict == "qualified"
            both_rescue += both and not default
            both_harm += default and not both
            both_only += both and not (default or first or second)
            if first_witness is None and verdicts[:3] == ["fidelity"] * 3 and both:
                first_witness = i
        n = len(outcomes[0])
        rows.append({
            **condition.fields(),
            "offered": n,
            "default_qualified": successes[0],
            "first_detour_qualified": successes[1],
            "second_detour_qualified": successes[2],
            "both_detours_qualified": successes[3],
            "both_rescue_vs_default": both_rescue,
            "both_harm_vs_default": both_harm,
            "only_both_detours_qualify": both_only,
        })
        if first_witness is None or condition.load != max(LOADS) or "seed" in witness:
            continue
        tape = tapes[condition.load][first_witness]
        witness = {
            "description": "Illustrative held-out replay where both detours are needed",
            "seed": tape.seed,
            "condition": condition.fields(),
            "planner_ms": planner_ms,
            "plans": {},
        }
        for plan in plans:
            trace = []
            result = simulate(plan, tape, planner_ms, trace, quantum_links=quantum_links)
            witness["plans"][plan.key] = {
                "quantum_nodes": plan.nodes,
                "control_routes": plan.outcome_routes,
                "verdict": condition.verdict(result),
                "fidelity": result.delivered_fidelity(condition.memory_scale),
                "wait_allowance_ms": result.wait_allowance_ms(condition.min_fidelity, condition.memory_scale),
                **asdict(result),
                "trace": trace,
            }
    return rows, witness


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    plans = plan_catalog()
    quantum_links = dict(QUANTUM)
    for edge in pairwise(QUANTUM_PATHS["B"]):
        quantum_links[edge] = replace(quantum_links[edge], success=args.secondary_success)
    conditions = [Condition(*values) for values in product(LOADS, MEMORY_SCALES, FIDELITIES)]
    cal_seeds, test_seeds = seed_lists(args.seed, args.calibration, args.holdout)

    start = perf_counter()
    cal_runs = {
        load: run_plans(plans, [make_workload(s, load) for s in cal_seeds], args.planner_ms, quantum_links)
        for load in LOADS
    }
    choices = {}
    cal_rows = []
    cal_metrics = {}
    for condition in conditions:
        metrics = {plan.key: measure(condition, cal_runs[condition.load][plan.key]) for plan in plans}
        cal_metrics[condition] = metrics
        for plan in plans:
            cal_rows.append({
                **condition.fields(),
                "plan": plan.key,
                "reserved_bytes_hops": plan.reserved_bytes_hops,
                "mean_screen_score": metrics[plan.key].mean_screen_score(condition),
                **metric_fields(metrics[plan.key]),
            })
        for budget in BUDGETS:
            choices[condition, budget] = choose(plans, metrics, condition, budget)
    calibration_s = perf_counter() - start

    tapes = {load: [make_workload(s, load) for s in test_seeds] for load in LOADS}
    test_runs = {load: run_plans(plans, t, args.planner_ms, quantum_links) for load, t in tapes.items()}
    test_rows = []
    plan_rows = []
    for condition in conditions:
        metrics = {plan.key: measure(condition, test_runs[condition.load][plan.key]) for plan in plans}
        for plan in plans:
            plan_rows.append({
                **condition.fields(),
                "plan": plan.key,
                "reserved_bytes_hops": plan.reserved_bytes_hops,
                **metric_fields(metrics[plan.key]),
            })
        for budget in BUDGETS:
            for name, plan in choices[condition, budget].items():
                test_rows.append({
                    **condition.fields(),
                    "budget_bytes_hops": budget,
                    "policy": name,
                    "plan": plan.key,
                    "reserved_bytes_hops": plan.reserved_bytes_hops,
                    "calibration_qualified_fraction": cal_metrics[condition][plan.key].qualified,
                    **metric_fields(metrics[plan.key]),
                })

    raw_rows = []
    for load in LOADS:
        for plan in plans:
            for seed, outcome in zip(test_seeds, test_runs[load][plan.key]):
                raw_rows.append({"load": load, "plan": plan.key, "seed": seed, **asdict(outcome)})
    aggregate = [
        paired_aggregate(conditions, choices, test_runs, budget, args.bootstrap)
        for budget in BUDGETS
    ]
    sensitivity = planner_sensitivity(
        conditions, choices, test_runs, tapes, args.planner_ms, quantum_links,
    )
    bundle_rows, witness = compare_bundles(conditions, test_runs, tapes, args.planner_ms, quantum_links)

    write_csv(args.output / "calibration.csv", cal_rows)
    write_csv(args.output / "heldout_by_plan.csv", plan_rows)
    write_csv(args.output / "heldout_by_policy.csv", test_rows)
    write_csv(args.output / "heldout_events.csv", raw_rows)
    write_csv(args.output / "planner_sensitivity.csv", sensitivity)
    write_csv(args.output / "bundle_comparison.csv", bundle_rows)
    write_json(args.output / "witness.json", witness)
    write_json(args.output / "manifest.json", {
        "master_seed": args.seed,
        "calibration_seeds": cal_seeds,
        "holdout_seeds": test_seeds,
        "planner_ms": args.planner_ms,
        "path_b_link_generation_success": args.secondary_success,
        "bootstrap_draws": args.bootstrap,
        "conditions": [c.fields() for c in conditions],
        "budgets_bytes_hops": BUDGETS,
        "plans": {
            p.key: {"quantum_nodes": p.nodes, "control_routes": p.outcome_routes,
                    "reserved_bytes_hops": p.reserved_bytes_hops}
            for p in plans
        },
        "quantum_links": {"->".join(edge): asdict(link) for edge, link in quantum_links.items()},
        "classical_links": {
            "->".join(edge): {"mbps": props[0], "propagation_ms": props[1]}
            for edge, props in CLASSICAL.items()
        },
        "activation_bytes": ACTIVATION_BYTES,
        "outcome_bytes": OUTCOME_BYTES,
        "background_packet_bytes": BACKGROUND_BYTES,
        "fifo_packets_including_service": QUEUE_PACKETS,
        "generation_attempt_limit": ATTEMPTS,
        "swap_duration_ms": SWAP_MS,
        "swap_success": SWAP_SUCCESS,
        "swap_visibility": SWAP_VISIBILITY,
        "background_warmup_ms": WARMUP_MS,
        "background_input_stops_ms": BACKGROUND_END_MS,
        "r3_to_b_load_as_fraction_of_r2_to_b": 0.35,
    })
    summary = {
        "calibration_seconds": calibration_s,
        "total_seconds": perf_counter() - start,
        "conditions": len(conditions),
        "path_b_link_generation_success": args.secondary_success,
        "holdout_seeds_per_condition": len(test_seeds),
        "calibration_seeds_per_condition": len(cal_seeds),
        "results_by_budget": aggregate,
        "planner_sensitivity": [row for row in sensitivity if row["scope"] == "all_conditions"],
        "witness_seed": witness.get("seed"),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected an integer greater than zero")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--calibration", type=positive_integer, default=400)
    parser.add_argument("--holdout", type=positive_integer, default=1200)
    parser.add_argument("--bootstrap", type=positive_integer, default=2000)
    parser.add_argument("--planner-ms", type=float, default=0.10)
    parser.add_argument("--secondary-success", type=float, default=0.85,
                        help="Elementary attempt success probability on both links of path B")
    args = parser.parse_args()
    if not 0 <= args.planner_ms <= DEADLINE_MS:
        parser.error("Planner time must be in the interval from zero to the application deadline")
    if not 0 < args.secondary_success <= 1:
        parser.error("Path B success probability must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
