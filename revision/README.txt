QControlFlow - independent-review revision
========================================

The original code/results at a928699 are unchanged. This directory adds a
separate, prespecified scientific revision. It does not establish that the
new controller outperforms sequential routing. The failed superiority
criterion is recorded in the evaluation summary and manuscript.

Method and evidence
-------------------
controller.py: vectorized whole-bundle forecast, shared outcome service,
              train-only monotone calibration, and compute-aware gate.
priority.py: finite nonpreemptive control-over-background priority.
runtime.py: profiled/measured clocks and bidirectional TRACK-inspired
            comparator. This is not a full reproduction of QNP.
protocol.json: graph/seed split, policies, estimand and success criterion.
results/profile.json: isolated timing profile and Apple M5 hardware record.
results/train: training data and frozen calibration (not evaluation data).
results/evaluate: full 14-graph, 6-seed primary campaign.
results/measured: serial measured-clock confirmation on 6 graphs.
results/ablation: decision-cost and outcome-coupling ablations.
results/probe: 504 fixed-emission self-contention replays.

Primary facts
-------------
There are 320 offers per evaluation run and a 64-offer burn-in. Ordinary
workloads have 107,520 retained offers per policy across 14 graphs. The
outage workload is reported separately. The policy rule and training model
were frozen in ac8850d and 0bf1de9 before held-out evaluation.

Controller-aware qualified delivery is 51.55%, versus shortest 50.04% and
factored sequential 51.69%. Its gain over shortest is 1.52 percentage points
(95% topology-bootstrap interval 0.93 to 2.11); the sequential comparison
is -0.14 points (-0.30 to 0.02). Serial confirmation instead finds a small
negative difference against sequential. The prespecified superiority
criterion is NOT met. Priority shortest-path service qualifies 76.74% but
changes the classical scheduling discipline; it is labeled separately.

Calibration is mixed: feedback reduces outage error substantially, but
ordinary Brier score and some outcome-count strata do not improve.
Within-request outcome interference changes qualification in 1/504 probe
cases. This probe freezes quantum emissions and is not a full traffic
interaction experiment.

Accounting and timing details
-----------------------------
Each policy has the same 3,000 planned byte-hop cap and per-node slot
capacity. Realized consumption is not forced equal. The paper's traffic
table uses whole-run control byte-hops divided by all offers, including
warm-up and telemetry. Delivery and slot-ms use retained requests.
The request-window traffic field in runs.csv is an auxiliary counter;
enqueue/service-start boundary conventions differ between the legacy and
priority engines, so the paper uses whole-tape totals for comparison.

The profiled full branch has an additional 0.02 ms calibration margin on
top of the common 0.02 ms dispatch allowance, as implemented in the frozen
profile code. This clarifies the brief budget-rule description in the
protocol without changing its frozen hash or the measured budgets.
The cost-aware post-search gain margin is a conservative heuristic; it is
not an optimal value-of-computation rule. Full cost is charged on fallback.

Reproduce from the repository root (Python 3.11+, NumPy required)
----------------------------------------------------------------
  python3 -m unittest revision.test_revision -v
  python3 -m revision.campaign --phase profile --output rerun/profile.json
  python3 -m revision.campaign --phase train --output rerun/train --profile rerun/profile.json --jobs 4
  python3 -m revision.campaign --phase evaluate --output rerun/evaluate --profile rerun/profile.json --calibration rerun/train/calibration.json --jobs 4
  python3 -m revision.campaign --phase measured --output rerun/measured --profile rerun/profile.json --calibration rerun/train/calibration.json --jobs 1
  python3 -m revision.campaign --phase ablation --output rerun/ablation --profile rerun/profile.json --calibration rerun/train/calibration.json --jobs 4
  python3 -m revision.self_contention --output rerun/probe --jobs 4
  python3 -m revision.analyze rerun/evaluate
  python3 -m revision.analyze rerun/measured
  python3 -m revision.analyze rerun/ablation

For a deterministic replay of the published profiled clocks, use the
published results/profile.json instead of generating a new host profile.
Measured clock results remain dependent on machine load and Python.
Do not run the serial timing confirmation alongside CPU-heavy tasks.

Large publication traces can be split losslessly:
  python3 -m revision.package_traces rerun/evaluate
  python3 -m revision.analyze rerun/evaluate
The analyzer reads either a single requests.csv.gz or manifest-listed
parts. Manifest hashes verify the split without changing any record.

Scope
-----
All traffic and physical parameters are synthetic. The 64-offer burn-in
does not prove steady state. The 14 graph instances, not individual request
records, are the bootstrap sampling units. Calibration uncertainty is not
included. Full QNP recovery, hardware validation, exact equality of consumed
resources, and a general superiority result remain open. No accessibility,
inclusion, or human-participant outcomes are inferred from these simulations.
