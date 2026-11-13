"""Run the paired experiments and export their request-level records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import gzip
from itertools import product
import json
from pathlib import Path
import platform
from random import Random
from statistics import mean, median
from time import perf_counter

import numpy as np

from network import Topology, edge_key, make_topology, paths, seed_for
from planner import Catalog
from simulator import Condition, Runtime, offer_requests

MAIN_POLICIES = ("qsp", "sequential", "mean", "single", "static", "joint", "chain")
MAIN_CONDITIONS = (
    Condition("light", 0.32, 1.00, 2.4),
    Condition("busy", 0.73, 1.00, 1.45),
    Condition("hot", 0.94, 1.00, 0.90),
    Condition("short_memory", 0.73, 0.60, 1.45),
    Condition("concurrent", 0.73, 1.00, 0.55),
)
STRESS_CONDITIONS = (
    Condition("outage", 0.78, 0.85, 1.35, outage=True),
    Condition("dephasing", 0.73, 1.00, 1.45, dephasing=True),
    Condition("cold", 0.73, 1.00, 1.45, cold=True),
    Condition("warm", 0.73, 1.00, 1.45),
)


@dataclass(frozen=True)
class Job:
    topology: str
    condition: Condition
    policies: tuple[str, ...]
    seeds: tuple[int, ...]
    requests: int
    fixed_planner_ms: float | None


def choose_pairs(topo: Topology) -> list[tuple[int, int]]:
    rng = Random(seed_for(topo.name, "pair catalog"))
    candidates = [(s, t) for s, t in product(topo.q_adj, repeat=2)
                  if s != t and edge_key(s, t) not in topo.q]
    rng.shuffle(candidates)
    chosen = []
    for s, t in candidates:
        route = paths(topo.q_adj, s, t, lambda *_: 1.0, 1, 4)
        if route and 3 <= len(route[0]) <= 5:
            chosen.append((s, t))
        if len(chosen) == 12:
            break
    if len(chosen) < 4:
        raise ValueError(f"Too few reachable candidate pairs in {topo.name}")
    return chosen


def quantile(values: list[float], p: float) -> float | None:
    return float(np.quantile(values, p)) if values else None


def aggregate(rows: list[dict], counters: dict) -> dict:
    successes = [r for r in rows if r["status"] == "qualified"]
    cpu = [r["selection_cpu_ms"] for r in rows if r["selection_cpu_ms"] > 0]
    delays = [r["terminal_ms"] for r in successes]
    offered = len(rows)
    admitted = sum(r["admitted"] for r in rows)
    run_ms = rows[-1]["offer_ms"] + 30 - rows[0]["offer_ms"]
    control = sum(v for k, v in counters["bytes_by_type"].items() if k != "background")
    return {
        "offered": offered,
        "admitted": admitted,
        "qualified": len(successes),
        "qualified_rate": len(successes) / offered,
        "qualified_per_s": 1000 * len(successes) / run_ms if run_ms else 0.0,
        "admitted_miss_rate": (admitted - len(successes)) / admitted if admitted else None,
        "p95_qualified_ms": quantile(delays, 0.95),
        "controller_median_ms": median(cpu) if cpu else None,
        "controller_p95_ms": quantile(cpu, 0.95),
        "control_bytes_hops": control,
        "telemetry_bytes_hops": counters["bytes_by_type"].get("telemetry", 0),
        "background_offered": counters["background_offered"],
        "background_delivered": counters["background_delivered"],
        "background_delivery_rate": counters["background_delivered"] / counters["background_offered"],
        "memory_slot_ms": sum(r["memory_slot_ms"] for r in rows),
        "peak_slots": counters["peak_used_slots"],
        "control_drops": sum(v for k, v in counters["drops_by_type"].items() if k != "background"),
    }


def run_job(job: Job) -> tuple[list[dict], list[dict], dict]:
    topo = make_topology(job.topology)
    catalog = Catalog(topo, job.condition.load)
    pairs = choose_pairs(topo)
    start = perf_counter()
    catalog.warm(pairs)
    compile_ms = 1000 * (perf_counter() - start)
    runs, requests = [], []
    for seed in job.seeds:
        offered = offer_requests(topo, pairs, job.condition, seed, job.requests)
        for policy in job.policies:
            run_keys = {"topology": job.topology, "condition": job.condition.name,
                        "seed": seed, "policy": policy}
            rows, counters = Runtime(catalog, job.condition, policy, seed, offered,
                                     job.fixed_planner_ms).run()
            runs.append({**run_keys, **aggregate(rows, counters)})
            requests.extend({**run_keys, **row, "physical_seed": row["seed"], "seed": seed}
                            for row in rows)
    return runs, requests, {
        "topology": job.topology, "condition": asdict(job.condition),
        "seeds": job.seeds, "policies": job.policies, "pairs": pairs,
        "nodes": len(topo.q_adj), "quantum_edges": len(topo.q),
        "classical_directed_edges": len(topo.c), "warm_catalog_build_ms": compile_ms,
        "candidate_plans": {f"{s}-{t}": len(catalog.table[s, t]) for s, t in pairs},
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.suffix == ".gz":
        stream = gzip.open(path, "wt", encoding="utf-8", newline="")
    else:
        stream = path.open("w", encoding="utf-8", newline="")
    with stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jobs_for(suite: str, seeds: int, requests: int,
             planner_ms: float | None) -> list[Job]:
    synthetic = tuple(f"synthetic-{size}-{i}" for size in (24, 36) for i in (1, 2, 3))
    data = Path(__file__).parent / "data"
    reference = tuple(name for name in ("abilene", "polska") if (data / f"{name}.json").exists())
    seed_list = tuple(2027 + n * 113 for n in range(seeds))
    if suite == "smoke":
        return [Job(synthetic[0], MAIN_CONDITIONS[1], MAIN_POLICIES + ("retry", "replicate"),
                    seed_list[:1], 12, planner_ms)]
    if suite == "main":
        return [Job(t, c, MAIN_POLICIES, seed_list, requests, planner_ms)
                for t, c in product(synthetic + reference, MAIN_CONDITIONS)]
    if suite == "confirm":
        new_seeds = tuple(4027 + n * 113 for n in range(seeds))
        policies = ("qsp", "sequential", "joint", "gated",
                    "fast_sequential", "fast_joint", "fast_gated")
        return [Job(t, c, policies, new_seeds, requests, planner_ms)
                for t, c in product(synthetic + reference, MAIN_CONDITIONS)]
    if suite == "reference":
        if len(reference) != 2:
            raise ValueError("Both reference adjacency files are required.")
        return [Job(t, c, MAIN_POLICIES, seed_list, requests, planner_ms)
                for t, c in product(reference, MAIN_CONDITIONS)]
    topologies = synthetic[:2] + synthetic[-2:] + reference
    cases = (
        (STRESS_CONDITIONS[0], ("sequential", "joint", "guarded", "replicate", "retry")),
        (STRESS_CONDITIONS[1], ("sequential", "joint")),
        (STRESS_CONDITIONS[2], ("sequential", "joint")),
        (STRESS_CONDITIONS[3], ("sequential", "joint")),
    )
    return [Job(t, c, policies, seed_list, requests, planner_ms)
            for t in topologies for c, policies in cases]


def sort_key(r: dict) -> tuple:
    return r["condition"], r["topology"], r["seed"], r["policy"], r.get("number", -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "main", "stress", "confirm", "reference"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--fixed-planner-ms", type=float)
    args = parser.parse_args()
    if min(args.jobs, args.seeds, args.requests) < 1:
        parser.error("Jobs, seeds and requests must be positive")
    if args.fixed_planner_ms is not None and args.fixed_planner_ms < 0:
        parser.error("Planner time must be nonnegative")
    jobs = jobs_for(args.suite, args.seeds, args.requests, args.fixed_planner_ms)
    args.output.mkdir(parents=True, exist_ok=True)
    all_runs, all_requests, settings = [], [], []
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as workers:
        futures = [workers.submit(run_job, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            runs, requests, setting = future.result()
            all_runs.extend(runs)
            all_requests.extend(requests)
            settings.append(setting)
            print(f"{n}/{len(jobs)} {setting['topology']} {setting['condition']['name']}", flush=True)
    write_csv(args.output / "runs.csv", sorted(all_runs, key=sort_key))
    write_csv(args.output / "requests.csv.gz", sorted(all_requests, key=sort_key))
    manifest = {
        "suite": args.suite, "python": platform.python_version(), "numpy": np.__version__,
        "cpu": platform.processor(), "workers": args.jobs, "latency_mode": (
            "measured inline" if args.fixed_planner_ms is None else args.fixed_planner_ms
        ),
        "runtime_s": perf_counter() - start, "runs": len(all_runs), "requests": len(all_requests),
        "jobs": sorted(settings, key=lambda j: (j["condition"]["name"], j["topology"])),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "jobs"}), flush=True)


if __name__ == "__main__":
    main()
