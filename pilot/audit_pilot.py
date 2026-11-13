"""Audit reported counts from the raw held-out events using direct fidelity checks."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from math import exp
from pathlib import Path

from run_pilot import boot_interval, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    stream = path.open(encoding="utf-8") if path.exists() else gzip.open(
        Path(f"{path}.gz"), "rt", encoding="utf-8"
    )
    with stream:
        return list(csv.DictReader(stream))


def condition(row: dict) -> tuple[float, float, float]:
    return tuple(float(row[k]) for k in ("load", "memory_scale", "min_fidelity"))


def passes(row: dict[str, str], memory_scale: float, minimum: float, deadline: float) -> bool:
    if row["failure"] or not row["frame_ready_ms"]:
        return False
    frame_time = float(row["frame_ready_ms"])
    if frame_time > deadline:
        return False
    gate_wait = frame_time - float(row["ready_ms"])
    exposure = float(row["pre_gate_exposure"]) + gate_wait * float(row["endpoint_rate"])
    visibility = float(row["raw_visibility"]) * exp(-exposure / memory_scale)
    return (1 + 3 * visibility) / 4 >= minimum


def audit_profile(folder: Path, exact_rows: list[dict[str, str]]) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    assert set(manifest["calibration_seeds"]).isdisjoint(manifest["holdout_seeds"])
    seed_order = [str(seed) for seed in manifest["holdout_seeds"]]
    rows = defaultdict(list)
    for row in read_csv(folder / "heldout_events.csv"):
        rows[float(row["load"]), row["plan"]].append(row)
    for group in rows.values():
        assert [row["seed"] for row in group] == seed_order
    checks = {}
    counts = {}
    for row in read_csv(folder / "heldout_by_plan.csv"):
        load, scale, minimum = cell = condition(row)
        plan = row["plan"]
        group = rows[load, plan]
        success = [passes(raw, scale, minimum, float(row["deadline_ms"])) for raw in group]
        checks[cell, plan] = success
        counts[cell, plan] = sum(success)
        assert len(success) == int(row["requests"])
        assert sum(success) == int(row["count_qualified"])
    for row in read_csv(folder / "heldout_by_policy.csv"):
        assert counts[condition(row), row["plan"]] == int(row["count_qualified"])
        assert int(row["reserved_bytes_hops"]) <= int(row["budget_bytes_hops"])

    cells = sorted(condition(row) for row in manifest["conditions"])
    results = {}
    for row in exact_rows:
        outcomes = {}
        for policy in ("sequential", "joint"):
            choices = row[f"{policy}_plans_in_cell_order"].split()
            reserve_control = sum(manifest["plans"][p]["reserved_bytes_hops"] for p in choices)
            reserve_slots = sum(2 * (len(manifest["plans"][p]["quantum_nodes"]) - 1) for p in choices)
            assert reserve_control / len(cells) == float(row["reserved_control_bytes_hops_per_request"])
            assert reserve_slots / len(cells) == float(row["reserved_memory_slots_per_request"])
            counts_for_policy = [counts[cell, p] for cell, p in zip(cells, choices, strict=True)]
            assert sum(counts_for_policy) / (len(cells) * len(seed_order)) == float(row[f"heldout_{policy}_fraction"])
            outcomes[policy] = [checks[cell, p] for cell, p in zip(cells, choices, strict=True)]
        if (row["conditions_using_path_a"], row["conditions_reserving_extra_hop"]) == ("6", "6"):
            differences = [
                (sum(joint) - sum(seq)) / len(cells)
                for joint, seq in zip(zip(*outcomes["joint"]), zip(*outcomes["sequential"]))
            ]
            results = {
                "reserved_control_bytes_hops_per_request": float(row["reserved_control_bytes_hops_per_request"]),
                "reserved_memory_slots_per_request": float(row["reserved_memory_slots_per_request"]),
                "sequential_fraction": float(row["heldout_sequential_fraction"]),
                "joint_fraction": float(row["heldout_joint_fraction"]),
                "difference_pp": float(row["heldout_difference_pp"]),
                "paired_seed_bootstrap_95_pp": boot_interval(differences, manifest["bootstrap_draws"]),
            }
    return {
        "raw_events": sum(map(len, rows.values())),
        "condition_plan_counts_checked": len(counts),
        "exact_reservations_checked": len(exact_rows),
        "midpoint_reservations": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    exact_rows = read_csv(args.results / "exact_reservation_comparison.csv")
    folders = sorted({row["profile"] for row in exact_rows})
    report = {
        folder: audit_profile(args.results / folder, [row for row in exact_rows if row["profile"] == folder])
        for folder in folders
    }
    write_json(args.results / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
