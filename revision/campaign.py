"""Frozen revision campaign, host profiling, and training-only calibration."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter, perf_counter_ns

import numpy as np

from campaign import MAIN_CONDITIONS, aggregate, choose_pairs, write_csv
from network import make_topology
from planner import Catalog
from simulator import Condition, offer_requests
from revision.controller import Calibrator, Controller
from revision.runtime import RevisionRuntime

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
CONDITIONS = MAIN_CONDITIONS + (Condition("outage", .78, .85, 1.35, outage=True),)


def hardware():
    def read(command):
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": platform.processor(),
            "cpu_model": read(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]),
            "logical_cpus": read(["/usr/sbin/sysctl", "-n", "hw.ncpu"]),
            "python": platform.python_version(), "numpy": np.__version__}


def profile(output):
    rows = []
    for size in [24, 36, 48]:
        topo = make_topology(f"synthetic-{size}-1")
        catalog = Catalog(topo, .73)
        pairs = choose_pairs(topo)[:4]
        catalog.warm(pairs)
        controller = Controller(catalog)
        seen = {edge: (0., 1.) for edge in topo.c}
        times = {p: [] for p in ["qsp", "fast_sequential", "fast_joint", "base", "full"]}
        for repetition in range(205):
            for s, d in pairs:
                for policy in times:
                    start = perf_counter_ns()
                    if policy in {"base", "full"}:
                        controller.decide(s, d, 24., .88, 1., topo.slots, 1., seen,
                                          "aware" if policy == "base" else "batch_joint",
                                          offer_rate=1e6 if policy == "base" else 0.)
                    else:
                        catalog.decide(s, d, 24., .88, 1., topo.slots, 1., seen, policy)
                    value = (perf_counter_ns() - start) / 1e6
                    if repetition >= 5:
                        times[policy].append(value)
        values = {p: {"median_ms": float(np.median(v)), "p95_ms": float(np.quantile(v, .95)),
                      "samples": len(v)} for p, v in times.items()}
        costs = {p: value["p95_ms"] + .02 for p, value in values.items()}
        costs["full"] = max(costs["base"], costs["full"] + .02)
        rows.append({"nodes": size, "measurements": values, "clock_costs_ms": costs})
    report = {"utc_time": datetime.now(timezone.utc).isoformat(), "hardware": hardware(),
              "protocol_sha256": sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
              "single_process": True, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def run_job(spec):
    graph, condition, policies, seeds, offers, burn, clock, profile_path, calibration_path = spec
    topo = make_topology(graph)
    catalog = Catalog(topo, condition.load)
    pairs = choose_pairs(topo)
    catalog.warm(pairs)
    profile_data = json.loads(Path(profile_path).read_text())
    size = min([24, 36, 48], key=lambda n: abs(n - len(topo.q_adj)))
    costs = next(r["clock_costs_ms"] for r in profile_data["rows"] if r["nodes"] == size)
    calibration = Calibrator(json.loads(Path(calibration_path).read_text())["model"]) if calibration_path else Calibrator()
    begin = perf_counter()
    controller = Controller(catalog, costs, calibration)
    build_ms = (perf_counter() - begin) * 1000
    all_runs, all_rows = [], []
    # Rotate policy order by seed to avoid a fixed timing-order advantage.
    for index, seed in enumerate(seeds):
        offered = offer_requests(topo, pairs, condition, seed, offers)
        order = policies[index % len(policies):] + policies[:index % len(policies)]
        for policy in order:
            runtime = RevisionRuntime(catalog, condition, policy, seed, offered,
                                      controller=controller, costs=costs, clock=clock, burn=burn)
            rows, counters = runtime.run()
            keys = {"topology": graph, "condition": condition.name, "seed": seed, "policy": policy}
            run = aggregate(rows, counters)
            selected = [r for r in rows if r["measurement"]]
            run.update({"measured_offered": len(selected),
                        "measured_qualified": sum(r["status"] == "qualified" for r in selected),
                        "measured_admitted": sum(r["admitted"] for r in selected),
                        "measured_control_byte_hops": run["control_bytes_hops"] - runtime.measurement_control_start,
                        "measured_memory_slot_ms": sum(r["memory_slot_ms"] for r in selected),
                        "clock": clock})
            all_runs.append({**keys, **run})
            all_rows.extend({**keys, **r, "physical_seed": r["seed"], "seed": seed} for r in rows)
    return all_runs, all_rows, {"graph": graph, "condition": asdict(condition),
                                "pairs": pairs, "coupled_catalog_build_ms": build_ms,
                                "nodes": len(topo.q_adj), "clock_costs_ms": costs}


def execute(args):
    protocol = PROTOCOL
    phase = args.phase
    if phase == "train":
        graphs = protocol["training_topologies"]
        seeds = protocol["training_seeds"]
        policies = protocol["training_policies"]
        offers, burn = protocol["training_offers"], protocol["training_burn_in"]
        conditions, clock = CONDITIONS, "profiled"
    else:
        graphs = protocol["evaluation_topologies"]
        seeds = protocol["evaluation_seeds"]
        policies = protocol["evaluation_policies"]
        offers, burn = protocol["evaluation_offers"], protocol["evaluation_burn_in"]
        conditions, clock = CONDITIONS, "profiled"
        if phase == "measured":
            graphs = [f"synthetic-{size}-{i}" for size in [24, 36, 48] for i in [4, 5]]
            seeds, policies, conditions, clock = seeds[:3], ["qsp", "fast_sequential", "fast_joint", "aware"], MAIN_CONDITIONS, "measured"
            if args.jobs != 1:
                raise ValueError("Measured confirmation must run with --jobs 1")
        if phase == "ablation":
            graphs = ["synthetic-24-6", "synthetic-24-7", "synthetic-36-6", "synthetic-36-7", "synthetic-48-6", "synthetic-48-7"]
            seeds, policies, conditions = seeds[:3], protocol["ablation_policies"], MAIN_CONDITIONS
    if args.smoke:
        graphs, conditions, seeds, offers, burn = graphs[:1], conditions[:1], seeds[:1], 24, 4
    if phase != "train" and not args.calibration:
        raise ValueError("Evaluation requires the frozen training calibration file")
    specs = [(g, c, list(policies), list(seeds), offers, burn, clock,
              str(args.profile.resolve()), str(args.calibration.resolve()) if args.calibration else None)
             for g in graphs for c in conditions]
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    results = []
    shards = args.output / "shards"
    shards.mkdir(exist_ok=True)

    def save_result(result):
        runs, rows, setting = result
        name = setting["condition"]["name"] + "__" + setting["graph"]
        target = shards / (name + ".csv.gz")
        key = lambda r: (r["seed"], r["policy"], r["number"])
        write_csv(target, sorted(rows, key=key))
        results.append((runs, target, setting))
    if args.jobs == 1:
        for i, spec in enumerate(specs, 1):
            result = run_job(spec)
            save_result(result)
            print(f"{phase}: {i}/{len(specs)} {spec[0]} {spec[1].name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_job, spec): spec for spec in specs}
            for i, future in enumerate(as_completed(futures), 1):
                save_result(future.result())
                spec = futures[future]
                print(f"{phase}: {i}/{len(specs)} {spec[0]} {spec[1].name}", flush=True)
    # Deterministic ordering regardless of job completion order.
    all_runs, settings = [], []
    for runs, _, setting in results:
        all_runs.extend(runs); settings.append(setting)
    sort_key = lambda r: (r["condition"], r["topology"], r["seed"], r["policy"], r.get("number", -1))
    write_csv(args.output / "runs.csv", sorted(all_runs, key=sort_key))
    count_rows = 0
    with gzip.open(args.output / "requests.csv.gz", "wt", newline="") as target:
        for i, (_, shard, _) in enumerate(sorted(results, key=lambda x: x[1].name)):
            with gzip.open(shard, "rt", newline="") as source:
                header = source.readline()
                if i == 0:
                    target.write(header)
                for line in source:
                    target.write(line)
                    count_rows += 1
    manifest = {"phase": phase, "smoke": args.smoke, "hardware": hardware(), "workers": args.jobs,
                "clock": clock, "offers_per_run": offers, "burn_in": burn, "runs": len(all_runs),
                "request_policy_records": count_rows, "graphs": graphs, "seeds": seeds,
                "policies": policies, "conditions": [c.name for c in conditions],
                "runtime_s": perf_counter() - start,
                "protocol_sha256": sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
                "profile_sha256": sha256(args.profile.read_bytes()).hexdigest(),
                "profiled_hardware": json.loads(args.profile.read_text())["hardware"],
                "calibration_sha256": sha256(args.calibration.read_bytes()).hexdigest() if args.calibration else None,
                "settings": sorted(settings, key=lambda s: (s["graph"], s["condition"]["name"]))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if phase == "train":
        def training_rows():
            with gzip.open(args.output / "requests.csv.gz", "rt", newline="") as source:
                for row in csv.DictReader(source):
                    if row["measurement"] != "True":
                        continue
                    row["admitted"] = row["admitted"] == "True"
                    row["raw_predicted"] = float(row["raw_predicted"]) if row["raw_predicted"] else None
                    yield row
        model = Calibrator.fit(training_rows())
        result = {"training_manifest_sha256": sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
                  "source": "training phase only; no evaluation records", "model": model.values}
        (args.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    for _, shard, _ in results:
        shard.unlink()
    shards.rmdir()
    print(json.dumps({k: v for k, v in manifest.items() if k != "settings"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["profile", "train", "evaluate", "measured", "ablation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=HERE / "results" / "profile.json")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "profile":
        profile(args.output)
    else:
        execute(args)


if __name__ == "__main__":
    main()
