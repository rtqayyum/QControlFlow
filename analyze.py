"""Audit the paired request traces and regenerate campaign tables and figures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from planner import BYTE_CAP

SUITES = ("main", "equal_clock", "stress", "confirm", "confirm_equal_clock")
CONDITIONS = ("light", "busy", "hot", "short_memory", "concurrent")
DISPLAY = {
    "light": "Light", "busy": "Busy", "hot": "Hot", "short_memory": "Short memory",
    "concurrent": "Concurrent",
}
IDENTITY = ("number", "source", "dest", "offer_ms", "deadline_ms", "min_fidelity", "physical_seed")
PAIR_SEED = 668727


def average(values):
    return float(np.mean(values)) if len(values) else None


def percentile(values, p):
    return float(np.quantile(values, p)) if len(values) else None


def key(row):
    return tuple(row[name] for name in ("condition", "topology", "seed", "policy"))


def load(path):
    manifest = json.loads((path / "manifest.json").read_text())
    with (path / "runs.csv").open(newline="") as stream:
        runs = {key(row): row for row in csv.DictReader(stream)}
    traces = defaultdict(list)
    with gzip.open(path / "requests.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            traces[key(row)].append(row)
    if set(runs) != set(traces) or len(runs) != manifest["runs"]:
        raise ValueError(f"Manifest and trace run sets differ in {path}")
    if sum(map(len, traces.values())) != manifest["requests"]:
        raise ValueError(f"Manifest and trace counts differ in {path}")

    pair_inputs = {}
    for run_key, rows in traces.items():
        rows.sort(key=lambda r: int(r["number"]))
        expected = list(range(int(runs[run_key]["offered"])))
        assert [int(r["number"]) for r in rows] == expected
        identity = tuple(tuple(row[name] for name in IDENTITY) for row in rows)
        assert identity == pair_inputs.setdefault(run_key[:3], identity)
        for r in rows:
            observed = r["status"] == "qualified"
            admitted = r["admitted"] == "True"
            realized = bool(r["fidelity"]) and (
                float(r["fidelity"]) >= float(r["min_fidelity"])
                and float(r["terminal_ms"]) <= float(r["deadline_ms"]) + 1e-9
            )
            assert observed == realized
            assert r["status"] != "pending"
            if r["fidelity"]:
                assert 0.25 <= float(r["fidelity"]) <= 1.0
                assert float(r["quantum_ready_ms"]) <= float(r["terminal_ms"]) + 1e-9
                assert float(r["storage_exposure"]) >= 0
            assert bool(r["route"]) == admitted
            assert bool(r["reserved_byte_hops"]) == admitted
            if admitted:
                assert 0 <= int(r["reserved_byte_hops"]) <= BYTE_CAP
        assert sum(r["status"] == "qualified" for r in rows) == int(runs[run_key]["qualified"])
        assert sum(r["admitted"] == "True" for r in rows) == int(runs[run_key]["admitted"])
    return manifest, runs, traces, pair_inputs


def metrics(rows, runs):
    qualified = [r for r in rows if r["status"] == "qualified"]
    admitted = [r for r in rows if r["admitted"] == "True"]
    delivered = [r for r in admitted if r["fidelity"]]
    cpu = [float(r["selection_cpu_ms"]) for r in rows if float(r["selection_cpu_ms"]) > 0]
    calibration = [r for r in admitted if r["predicted"]]
    high = [r for r in calibration if float(r["predicted"]) >= .8]
    n = len(rows)
    control = sum(int(r["control_bytes_hops"]) for r in runs)
    telemetry = sum(int(r["telemetry_bytes_hops"]) for r in runs)
    bg_in = sum(int(r["background_offered"]) for r in runs)
    bg_out = sum(int(r["background_delivered"]) for r in runs)
    return {
        "offered": n,
        "qualified": len(qualified),
        "qualified_pct": 100 * len(qualified) / n,
        "admitted": len(admitted),
        "admitted_miss_pct": 100 * (len(admitted) - len(qualified)) / len(admitted) if admitted else None,
        "qualified_pairs_per_s": average([float(r["qualified_per_s"]) for r in runs]),
        "p95_qualified_ms": percentile([float(r["terminal_ms"]) for r in qualified], .95),
        "p50_cpu_ms": percentile(cpu, .50),
        "p95_cpu_ms": percentile(cpu, .95),
        "mean_qualified_fidelity": average([float(r["fidelity"]) for r in qualified]),
        "p95_controller_wait_ms": percentile([float(r["controller_wait_ms"]) for r in rows], .95),
        "p95_frame_wait_ms": percentile(
            [float(r["terminal_ms"]) - float(r["quantum_ready_ms"]) for r in delivered], .95
        ),
        "control_byte_hops_per_offer": control / n,
        "telemetry_byte_hops_per_offer": telemetry / n,
        "reserved_byte_hops_per_admitted": average([int(r["reserved_byte_hops"]) for r in admitted]),
        "memory_slot_ms_per_offer": sum(float(r["memory_slot_ms"]) for r in rows) / n,
        "background_delivery_pct": 100 * bg_out / bg_in,
        "calibration": {
            "brier": average([(float(r["predicted"]) - (r["status"] == "qualified")) ** 2 for r in calibration]),
            "high_count": len(high),
            "high_predicted": average([float(r["predicted"]) for r in high]),
            "high_observed": average([r["status"] == "qualified" for r in high]),
        },
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }


def summarize(runs, traces):
    grouped_runs = defaultdict(list)
    grouped_traces = defaultdict(list)
    for k, values in traces.items():
        condition, _, _, policy = k
        for group in (condition, "all"):
            grouped_runs[group, policy].append(runs[k])
            grouped_traces[group, policy].extend(values)
    groups = defaultdict(dict)
    for (condition, policy), rows in grouped_traces.items():
        groups[condition][policy] = metrics(rows, grouped_runs[condition, policy])
    return groups


def paired(traces, condition, policy, baseline, reference_condition=None, topology_names=None):
    differences = defaultdict(list)
    changes = paired_admitted = only_policy = only_baseline = 0
    for (c, topology, seed, p), rows in traces.items():
        if p != policy or (condition != "all" and c != condition):
            continue
        if topology_names is not None and topology not in topology_names:
            continue
        old = traces.get((reference_condition or c, topology, seed, baseline))
        if old is None:
            continue
        for a, b in zip(rows, old, strict=True):
            assert all(a[name] == b[name] for name in IDENTITY)
            improved = a["status"] == "qualified" and b["status"] != "qualified"
            harmed = b["status"] == "qualified" and a["status"] != "qualified"
            only_policy += improved
            only_baseline += harmed
            differences[topology].append(int(improved) - int(harmed))
            if a["admitted"] == b["admitted"] == "True":
                paired_admitted += 1
                changes += a["route"] != b["route"]
    by_graph = {graph: 100 * average(values) for graph, values in sorted(differences.items())}
    values = np.array(list(by_graph.values()))
    rng = np.random.default_rng(PAIR_SEED)
    bootstrap = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return {
        "topologies": len(values),
        "pairs": sum(map(len, differences.values())),
        "delta_pp": float(values.mean()),
        "ci_pp": list(map(float, np.quantile(bootstrap, [.025, .975]))),
        "by_topology_pp": by_graph,
        "only_policy": only_policy,
        "only_baseline": only_baseline,
        "changed_quantum_path": changes,
        "both_admitted": paired_admitted,
    }


def compare(suite, traces):
    if suite == "stress":
        cases = [
            ("outage", p, "sequential", None) for p in ("joint", "guarded", "replicate", "retry")
        ] + [
            ("dephasing", "joint", "sequential", None),
            ("cold", "joint", "joint", "warm"),
            ("cold", "sequential", "sequential", "warm"),
        ]
    else:
        policies = (("joint", "sequential"),)
        if suite.startswith("confirm"):
            policies += (("gated", "sequential"), ("fast_joint", "fast_sequential"),
                         ("fast_gated", "fast_sequential"), ("fast_joint", "joint"),
                         ("fast_gated", "gated"), ("fast_sequential", "sequential"))
        cases = [(c, p, b, None) for c in (*CONDITIONS, "all") for p, b in policies]
    return [
        {"condition": c, "policy": p, "baseline": b, "baseline_condition": bc or c,
         **paired(traces, c, p, b, bc)}
        for c, p, b, bc in cases
    ]


def plot(summary, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 7.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), layout="constrained",
                             gridspec_kw={"width_ratios": [1.2, 1]})
    left, right = axes
    positions = np.arange(len(CONDITIONS))
    for suite, color, offset, label in (
        ("main", "#8b402a", -.12, "Measured time"),
        ("equal_clock", "#12667e", .12, "Fixed 0.30 ms"),
    ):
        rows = {r["condition"]: r for r in summary["effects"][suite]}
        effect = np.array([rows[c]["delta_pp"] for c in CONDITIONS])
        ci = np.array([rows[c]["ci_pp"] for c in CONDITIONS])
        left.errorbar(effect, positions + offset, xerr=[effect - ci[:, 0], ci[:, 1] - effect],
                      fmt="o", color=color, capsize=2.5, markersize=3.5, label=label)
    left.axvline(0, color="#767676", linewidth=.7)
    left.set(yticks=positions, yticklabels=[DISPLAY[c] for c in CONDITIONS],
             xlabel="Joint minus sequential (percentage points)")
    left.invert_yaxis()
    left.legend(loc="upper left", frameon=False)
    comparisons = ("sequential", "joint", "gated")
    group = summary["metrics"]["confirm"]["all"]
    for i, p in enumerate(comparisons):
        for j, policy in enumerate((p, f"fast_{p}")):
            m = group[policy]
            right.barh(i + (.16 if j else -.16), m["p50_cpu_ms"], height=.29,
                       color=("#8b402a", "#12667e")[j],
                       label=("Direct score", "Factored score")[j] if i == 0 else None)
            right.plot(m["p95_cpu_ms"], i + (.16 if j else -.16), marker="|",
                       markersize=7, color="#333333")
    right.set(yticks=range(3), yticklabels=["Sequential", "Joint", "Gated"],
              xlabel="Selection time (ms)")
    right.invert_yaxis()
    right.legend(loc="lower right", frameon=False)
    left.set_title("(a) End-to-end qualified delivery")
    right.set_title("(b) Held-out selection time")
    fig.savefig(target / "controller.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.28), layout="constrained")
    pilot_summary = Path(__file__).parent / "pilot/results/reference/bundle_comparison.csv"
    with pilot_summary.open(newline="") as stream:
        pilot = next(r for r in csv.DictReader(stream) if (
            float(r["load"]), float(r["memory_scale"]), float(r["min_fidelity"])
        ) == (.9, 1.0, .85))
    successes = tuple(int(pilot[f"{name}_qualified"])
                      for name in ("default", "first_detour", "second_detour", "both_detours"))
    offers = int(pilot["offered"])
    labels = ("Default", "First", "Second", "Both")
    colors = ("#6a747d", "#7f9299", "#7f9299", "#12667e")
    bars = ax.bar(labels, [100 * n / offers for n in successes], color=colors, width=.62)
    ax.bar_label(bars, labels=[str(n) for n in successes], padding=3)
    ax.set(ylim=(0, 104), ylabel="Qualified delivery (%)", xlabel="Outcome routes changed")
    ax.set_yticks((0, 25, 50, 75, 100))
    fig.savefig(target / "pilot.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    summary = {"counts": {}, "metrics": {}, "effects": {}, "reference_effects": {}}
    identities = {}
    for suite in SUITES:
        manifest, runs, traces, inputs = load(args.results / suite)
        identities[suite] = inputs
        summary["counts"][suite] = {
            name: manifest[name] for name in ("runs", "requests", "latency_mode")
        }
        summary["metrics"][suite] = summarize(runs, traces)
        summary["effects"][suite] = compare(suite, traces)
        if suite != "stress":
            summary["reference_effects"][suite] = {
                c: paired(traces, c, "joint", "sequential", topology_names={"abilene", "polska"})
                for c in CONDITIONS
            }
        if suite == "confirm_equal_clock":
            fields = set(next(iter(traces.values()))[0]) - {"policy", "selection_cpu_ms"}
            for (condition, graph, seed, policy), rows in traces.items():
                if policy.startswith("fast_"):
                    direct = traces[condition, graph, seed, policy.removeprefix("fast_")]
                    for a, b in zip(rows, direct, strict=True):
                        assert all(a[field] == b[field] for field in fields)
    assert identities["main"] == identities["equal_clock"]
    assert identities["confirm"] == identities["confirm_equal_clock"]
    assert not set(identities["main"]).intersection(identities["confirm"])
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.results / "paired_effects.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("suite", "condition", "policy", "baseline", "baseline_condition", "pairs",
                         "topologies", "delta_pp", "lower_pp", "upper_pp", "only_policy", "only_baseline",
                         "both_admitted", "changed_quantum_path"))
        for suite, comparisons in summary["effects"].items():
            for e in comparisons:
                writer.writerow((suite, e["condition"], e["policy"], e["baseline"], e["baseline_condition"],
                                 e["pairs"], e["topologies"], e["delta_pp"], *e["ci_pp"],
                                 e["only_policy"], e["only_baseline"], e["both_admitted"],
                                 e["changed_quantum_path"]))
    if not args.no_figures:
        plot(summary, args.figures)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
