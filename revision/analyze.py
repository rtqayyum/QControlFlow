"""Audit held-out revision traces and compute paired effects and calibration."""
from collections import Counter, defaultdict
from hashlib import sha256
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from planner import BYTE_CAP


class CalibrationStats:
    def __init__(self):
        self.n = self.brier = self.sum_p = self.sum_y = 0
        self.bins = np.zeros((10, 3))

    def add(self, p, y):
        assert 0 <= p <= 1
        self.n += 1
        self.brier += (p - y) ** 2
        self.sum_p += p
        self.sum_y += y
        self.bins[min(9, int(p * 10))] += [1, p, y]

    def result(self):
        bins = [{"lower": i / 10, "upper": (i + 1) / 10, "n": int(v[0]),
                 "predicted": float(v[1] / v[0]), "observed": float(v[2] / v[0])}
                for i, v in enumerate(self.bins) if v[0]]
        return {"n": self.n, "brier": self.brier / self.n,
                "mean_predicted": self.sum_p / self.n, "observed": self.sum_y / self.n,
                "ece_10": sum(abs(v[1] - v[2]) for v in self.bins) / self.n,
                "bins": bins}


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    outcomes = defaultdict(list)
    identities = defaultdict(lambda: sha256())
    counts = defaultdict(Counter)
    values = defaultdict(lambda: defaultdict(list))
    status = defaultdict(Counter)
    calibration = defaultdict(CalibrationStats)
    total = 0
    with gzip.open(folder / "requests.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            c, g, seed, policy = (row[x] for x in ["condition", "topology", "seed", "policy"])
            identity = tuple(row[x] for x in ["number", "source", "dest", "offer_ms", "deadline_ms", "min_fidelity", "physical_seed"])
            identities[c, g, seed, policy].update(repr(identity).encode())
            admitted = row["admitted"] == "True"
            qualified = row["status"] == "qualified"
            realized = bool(row["fidelity"]) and float(row["fidelity"]) >= float(row["min_fidelity"]) and float(row["terminal_ms"]) <= float(row["deadline_ms"]) + 1e-9
            assert qualified == realized
            assert row["status"] != "pending"
            if admitted:
                assert 0 <= int(row["reserved_byte_hops"]) <= BYTE_CAP
            if row["measurement"] != "True":
                continue
            outcomes[c, g, seed, policy].append(int(qualified))
            for group in [c, "ordinary" if c != "outage" else "outage_only"]:
                key = group, policy
                counts[key].update(offered=1, admitted=int(admitted), qualified=int(qualified))
                status[key][row["status"]] += 1
                for field in ["selection_cpu_ms", "controller_wait_ms", "memory_slot_ms"]:
                    if row[field]: values[key][field].append(float(row[field]))
                if row["selection_applied_ms"]:
                    values[key]["selection_applied_ms"].append(float(row["selection_applied_ms"]))
                if admitted:
                    values[key]["reserved_byte_hops"].append(int(row["reserved_byte_hops"]))
                    counts[key]["mode_" + str(row["decision_mode"])] += 1
                if qualified:
                    values[key]["delivery_ms"].append(float(row["terminal_ms"]))
            if admitted:
                groups = [("condition", c), ("memory_scale", row["memory_scale"]), ("outcomes", row["outcomes"])]
                if c != "outage": groups.append(("aggregate", "ordinary"))
                for which in ["raw_predicted", "calibrated_predicted", "predicted"]:
                    if row[which] == "": continue
                    for dimension, value in groups:
                        calibration[policy, dimension, value, which].add(float(row[which]), int(qualified))
    assert total == manifest["request_policy_records"]
    reference = {}
    for key, value in identities.items():
        short = key[:3]
        digest = value.hexdigest()
        assert reference.setdefault(short, digest) == digest
    with (folder / "runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    assert len(runs) == manifest["runs"]
    services = defaultdict(float)
    for run in runs:
        for group in [run["condition"], "ordinary" if run["condition"] != "outage" else "outage_only"]:
            services[group, run["policy"]] += int(run["measured_control_byte_hops"])
    metrics = defaultdict(dict)
    for (group, policy), counter in counts.items():
        v = values[group, policy]
        q = lambda field, percentile: float(np.quantile(v[field], percentile)) if v[field] else None
        offered = counter["offered"]
        metrics[group][policy] = {**dict(counter), "qualified_pct": 100 * counter["qualified"] / offered,
            "admitted_miss_pct": 100 * (counter["admitted"] - counter["qualified"]) / counter["admitted"],
            "control_byte_hops_per_offer": services[group, policy] / offered,
            "memory_slot_ms_per_offer": sum(v["memory_slot_ms"]) / offered,
            "mean_reserved_byte_hops": float(np.mean(v["reserved_byte_hops"])),
            "p50_cpu_ms": q("selection_cpu_ms", .5), "p95_cpu_ms": q("selection_cpu_ms", .95),
            "p50_applied_ms": q("selection_applied_ms", .5),
            "p95_controller_wait_ms": q("controller_wait_ms", .95),
            "p95_delivery_ms": q("delivery_ms", .95), "statuses": dict(status[group, policy])}
    policies = manifest["policies"]
    comparisons = []
    pairs = [("aware", p) for p in policies if p != "aware"]
    if "batch_joint" in policies:
        pairs += [("batch_joint", "qsp"), ("batch_joint", "fast_sequential")]
    groups = list(manifest["conditions"]) + ["ordinary"]
    for group in groups:
        for policy, baseline in pairs:
            diffs = defaultdict(list)
            for (c, graph, seed, p), ys in outcomes.items():
                if p != policy or (c != group and not (group == "ordinary" and c != "outage")):
                    continue
                old = outcomes[c, graph, seed, baseline]
                assert len(ys) == len(old)
                diffs[graph].extend(a - b for a, b in zip(ys, old))
            if not diffs: continue
            by_graph = {g: 100 * float(np.mean(d)) for g, d in sorted(diffs.items())}
            data = np.array(list(by_graph.values()))
            rng = np.random.default_rng(795027)
            boot = rng.choice(data, (10000, len(data)), replace=True).mean(axis=1)
            comparisons.append({"condition": group, "policy": policy, "baseline": baseline,
                                "paired_offers": sum(map(len, diffs.values())), "graphs": len(data),
                                "delta_pp": float(data.mean()), "ci_pp": list(np.quantile(boot, [.025, .975])),
                                "by_graph_pp": by_graph})
    cal = [{"policy": key[0], "dimension": key[1], "value": key[2], "forecast": key[3], **s.result()}
           for key, s in sorted(calibration.items()) if s.n]
    success = []
    for baseline in ["qsp", "fast_sequential"]:
        result = next((x for x in comparisons if x["condition"] == "ordinary" and x["policy"] == "aware" and x["baseline"] == baseline), None)
        if result is not None: success.append(result["ci_pp"][0] > 0)
    result = {"manifest": {k: v for k, v in manifest.items() if k != "settings"}, "metrics": metrics,
              "effects": comparisons, "calibration": cal,
              "prespecified_superiority_criterion_met": len(success) == 2 and all(success),
              "audit": {"request_records_checked": total, "policy_tapes_identical": True,
                        "qualification_and_budget_checks": True}}
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = summarize(args.folder)
    print(json.dumps({"audit": report["audit"],
                      "ordinary": report["metrics"].get("ordinary"),
                      "effects": [r for r in report["effects"] if r["condition"] == "ordinary"],
                      "superiority_criterion_met": report["prespecified_superiority_criterion_met"]}, indent=2))


if __name__ == "__main__":
    main()
