"""Compare exact control and memory reservations using the completed pilot runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pilot_model import OUTCOME_BYTES, Plan, plan_catalog
from run_pilot import write_csv

CATALOG = plan_catalog()
PLAN_BY_KEY = {plan.key: plan for plan in CATALOG}
DEFAULTS = {plan.quantum_path: plan for plan in CATALOG if plan.uses_default_routes}


def extra_hops(plan: Plan) -> int:
    added = plan.reserved_bytes_hops - DEFAULTS[plan.quantum_path].reserved_bytes_hops
    hops, remainder = divmod(added, OUTCOME_BYTES)
    if hops not in (0, 1) or remainder:
        raise ValueError("The exact matcher expects zero or one extra control hop per plan.")
    return hops


def load_counts(path: Path) -> tuple[dict[tuple[float, float, float], dict[str, int]], int]:
    result = {}
    sizes = set()
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cell = tuple(float(row[k]) for k in ("load", "memory_scale", "min_fidelity"))
            result.setdefault(cell, {})[row["plan"]] = int(row["count_qualified"])
            sizes.add(int(row["requests"]))
    if len(sizes) != 1 or any(set(plans) != set(PLAN_BY_KEY) for plans in result.values()):
        raise ValueError(f"Incomplete plan catalog or unequal request counts in {path}")
    return result, sizes.pop()


def quantum_first(calibration: list[dict[str, int]], a_count: int) -> tuple[str, ...]:
    # All plans on a given quantum path use the same memory and base control bytes.
    ranked = sorted(
        range(len(calibration)),
        key=lambda i: (-(calibration[i]["A-00"] - calibration[i]["B-0"]), i),
    )
    a_cells = set(ranked[:a_count])
    return tuple("A" if i in a_cells else "B" for i in range(len(calibration)))


def sequential_choices(
    calibration: list[dict[str, int]], a_count: int, extra_count: int,
) -> tuple[str, ...]:
    paths = quantum_first(calibration, a_count)
    choices = []
    for path, counts in zip(paths, calibration, strict=True):
        by_budget = []
        for extra in (0, 1):
            options = [p for p in CATALOG if p.quantum_path == path and extra_hops(p) == extra]
            by_budget.append(min(options, key=lambda p: (-counts[p.key], p.key)).key)
        choices.append(tuple(by_budget))
    ranked = sorted(
        range(len(calibration)),
        key=lambda i: (-(calibration[i][choices[i][1]] - calibration[i][choices[i][0]]), i),
    )
    extra_cells = set(ranked[:extra_count])
    return tuple(plans[int(i in extra_cells)] for i, plans in enumerate(choices))


def joint_choices(calibration: list[dict[str, int]]) -> dict[tuple[int, int], tuple[str, ...]]:
    states: dict[tuple[int, int], tuple[int, tuple[str, ...]]] = {(0, 0): (0, ())}
    for counts in calibration:
        next_states = {}
        for (a_used, extra_used), (score, history) in states.items():
            for plan in CATALOG:
                key = (a_used + (plan.quantum_path == "A"), extra_used + extra_hops(plan))
                total = score + counts[plan.key]
                current = next_states.get(key)
                if current is None or total > current[0]:
                    next_states[key] = (total, history + (plan.key,))
        states = next_states
    return {key: history for key, (_, history) in states.items()}


def reserved(keys: tuple[str, ...]) -> tuple[int, int]:
    plans = [PLAN_BY_KEY[key] for key in keys]
    return sum(p.reserved_bytes_hops for p in plans), sum(2 * (len(p.nodes) - 1) for p in plans)


def compare(folder: Path) -> list[dict]:
    training, n_train = load_counts(folder / "calibration.csv")
    holdout, n_test = load_counts(folder / "heldout_by_plan.csv")
    if training.keys() != holdout.keys():
        raise ValueError("Calibration and holdout conditions differ.")
    cells = sorted(training)
    cal = [training[c] for c in cells]
    test = [holdout[c] for c in cells]
    rows = []
    for (a_count, extra_count), joint in sorted(joint_choices(cal).items()):
        sequential = sequential_choices(cal, a_count, extra_count)
        control, slots = reserved(joint)
        assert (control, slots) == reserved(sequential)
        train_joint = sum(counts[key] for counts, key in zip(cal, joint, strict=True))
        train_seq = sum(counts[key] for counts, key in zip(cal, sequential, strict=True))
        assert train_joint >= train_seq
        test_joint = sum(counts[key] for counts, key in zip(test, joint, strict=True))
        test_seq = sum(counts[key] for counts, key in zip(test, sequential, strict=True))
        offered = n_test * len(cells)
        rows.append({
            "profile": folder.name,
            "conditions_using_path_a": a_count,
            "conditions_reserving_extra_hop": extra_count,
            "reserved_control_bytes_hops_per_request": control / len(cells),
            "reserved_memory_slots_per_request": slots / len(cells),
            "heldout_requests_per_policy": offered,
            "calibration_joint_fraction": train_joint / (n_train * len(cells)),
            "calibration_sequential_fraction": train_seq / (n_train * len(cells)),
            "heldout_joint_fraction": test_joint / offered,
            "heldout_sequential_fraction": test_seq / offered,
            "heldout_difference_pp": 100 * (test_joint - test_seq) / offered,
            "joint_plans_in_cell_order": " ".join(joint),
            "sequential_plans_in_cell_order": " ".join(sequential),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_csv(args.output, [row for folder in args.profiles for row in compare(folder)])


if __name__ == "__main__":
    main()
