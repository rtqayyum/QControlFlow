"""Causal packet replay with frozen quantum emissions and shared background."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise
import json
from math import exp, log, isfinite
from pathlib import Path

import numpy as np

from campaign import choose_pairs, write_csv
from network import Network, Packet, TRACK_BYTES, make_topology, seed_for
from planner import Catalog
from revision.controller import coupled_arrivals

PROTOCOL = json.loads((Path(__file__).parent / "protocol.json").read_text())


def replay(topo, controls, emission, load, seed, include):
    net = Network(topo, None)
    net.load_background(load, seed, 40.)
    arrival = {}
    for i in include:
        callback = lambda t, i=i: arrival.update({i: t})
        net.at(float(emission[i]), net.send, Packet(controls[i], TRACK_BYTES, "outcome", callback))
    net.run()
    return arrival


def run_case(graph):
    topo = make_topology(graph)
    rows = []
    for load in [.32, .73, .94]:
        catalog = Catalog(topo, load)
        pairs = choose_pairs(topo)
        catalog.warm(pairs)
        candidates = []
        for pair in pairs:
            for plan in catalog.table[pair]:
                if plan.detours or len(plan.controls) < 2: continue
                edges = [list(pairwise(r)) for r in plan.controls]
                shared = sum(len(set(a) & set(b)) for i, a in enumerate(edges) for b in edges[i + 1:])
                if shared: candidates.append((pair, plan, shared))
        if not candidates:
            continue
        for case in range(12):
            pair, plan, shared = candidates[case % len(candidates)]
            j = case % catalog.n
            emission = np.array([plan.arrivals[i, j] - catalog.delay(r, TRACK_BYTES)[j]
                                 for i, r in enumerate(plan.controls)])
            seed = seed_for("revision-self-contention", graph, load, case)
            joint = replay(topo, plan.controls, emission, load, seed, range(len(plan.controls)))
            isolated = {}
            for i in range(len(plan.controls)):
                isolated.update(replay(topo, plan.controls, emission, load, seed, [i]))
            tj = max(joint.values()) if len(joint) == len(plan.controls) else None
            ti = max(isolated.values()) if len(isolated) == len(plan.controls) else None
            q = float(plan.ready[j])
            cutoff = min(24., q + (plan.log_raw - log((4 * .88 - 1) / 3) - plan.pre_exposure[j]) / plan.endpoint_rate)
            independent_prediction = float(max(q, plan.arrivals[:, j].max()))
            expected_coupled = coupled_arrivals(topo, plan.controls,
                np.array([plan.arrivals[i] - catalog.delay(r, TRACK_BYTES)
                          for i, r in enumerate(plan.controls)]), catalog.waits)
            rows.append({"topology": graph, "load": load, "case": case, "seed": seed,
                         "outcomes": len(plan.controls), "shared_edge_pairs": shared,
                         "quantum_ready_ms": q, "cutoff_ms": cutoff,
                         "joint_frame_ms": max(q, tj) if tj is not None else None,
                         "isolated_frame_ms": max(q, ti) if ti is not None else None,
                         "joint_qualified": tj is not None and max(q, tj) <= cutoff,
                         "isolated_qualified": ti is not None and max(q, ti) <= cutoff,
                         "independent_forecast_ms": independent_prediction,
                         "coupled_forecast_ms": float(max(q, expected_coupled[:, j].max()))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = {pool.submit(run_case, graph): graph for graph in PROTOCOL["evaluation_topologies"]}
        for future in as_completed(tasks):
            rows.extend(future.result())
            print("probe:", tasks[future], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["topology"], r["load"], r["case"]))
    write_csv(args.output / "self_contention.csv", rows)
    differences = [r["joint_frame_ms"] - r["isolated_frame_ms"] for r in rows
                   if r["joint_frame_ms"] is not None and r["isolated_frame_ms"] is not None]
    summary = {"cases": len(rows), "quantum_emissions_frozen": True,
               "identical_background_tapes": True,
               "qualification_disagreements": int(sum(r["joint_qualified"] != r["isolated_qualified"] for r in rows)),
               "both_complete_cases": len(differences),
               "median_joint_minus_isolated_ms": float(np.median(differences)),
               "p95_joint_minus_isolated_ms": float(np.quantile(differences, .95)),
               "max_joint_minus_isolated_ms": float(max(differences)),
               "joint_only_missing_outcome": sum(r["joint_frame_ms"] is None and r["isolated_frame_ms"] is not None for r in rows)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
