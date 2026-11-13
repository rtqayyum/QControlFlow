# QControlFlow five-node co-routing pilot

This independent CPU experiment tests whether the quantum route and the routes for classical swap outcomes should be selected jointly. It implements the first pilot in the QControlFlow revision plan. The original Figure 3 results require the original simulator and configurations.

## Main findings

The application accepts a pair if its frame is available by 10 ms and its fidelity meets the requested threshold. Each physical profile uses 12 conditions with 400 independent calibration seeds and 1,200 held-out seeds per condition. The same seeds are replayed across policies and conditions. Each policy is evaluated on 14,400 offered requests per profile. The sampling unit for a paired interval is the seed.

All four policies picked the same plan in the reference profile. The initial and final reference runs produced byte-identical held-out event records. The follow-up sweep changed only the elementary generation success probability on both links of quantum path B and used fresh calibration and holdout seeds.

| Path B success per attempt | Sequential qualified | Joint qualified | Joint gain in percentage points | Paired 95% interval |
|---|---:|---:|---:|---:|
| 0.55 | 80.90% | 83.40% | +2.50 | +1.73 to +3.22 |
| 0.65 | 84.26% | 85.58% | +1.31 | +0.80 to +1.78 |
| 0.75 | 88.51% | 88.47% | −0.03 | −0.24 to +0.15 |
| 0.85 | 92.55% | 92.55% | 0.00 | 0.00 to 0.00 |

The table uses a common per-request reservation cap of 576 control byte-hops. The positive cases used more control traffic on average. At 0.55, the mean was 511.83 byte-hops for joint routing and 392.55 for sequential. At 0.65, the corresponding means were 402.67 and 314.51. At the tighter per-request cap of 480, joint and sequential selected the same plans throughout. Both detours on path A require 576 byte-hops in the pilot.

At the exploratory **exact average reservation** of 400 control byte-hops and 5 modeled memory slots per request, the policies reserved the same total control and memory resources across the 12 equally weighted conditions. They could allocate those resources to different conditions.

| Profile | Sequential qualified | Joint qualified | Joint gain in percentage points | Paired 95% interval |
|---|---:|---:|---:|---:|
| Reference | 89.11% | 90.91% | +1.80 | +1.49 to +2.09 |
| Sweep 0.55 | 79.76% | 81.82% | +2.06 | +1.76 to +2.34 |
| Sweep 0.65 | 83.76% | 85.26% | +1.50 | +1.27 to +1.74 |
| Sweep 0.75 | 86.86% | 88.08% | +1.22 | +0.88 to +1.53 |
| Sweep 0.85 | 89.25% | 90.84% | +1.59 | +1.27 to +1.90 |

This reservation comparison is a different resource constraint from the per-request cap. At the diagnostic point, each policy uses the three-link quantum path in six conditions and reserves an extra control hop in six conditions. The remaining conditions use the two-link path or the minimum control-hop reservation. The exact match is on planned resources. Individual attempts can use fewer bytes after an early failure, and the slot count is a reservation proxy for the number of elementary memory halves. The bootstrap intervals describe held-out seed variation within the synthetic model and keep the calibrated plans fixed.

### Why coordinating both messages matters

At load 0.90, baseline memory lifetime and minimum fidelity 0.85 in the reference run, quantum path A delivered 571 of 1,200 qualified pairs with default forwarding. Detouring only the first outcome delivered 593, and detouring only the second delivered 591. Detouring both delivered 1,053. There were **443 paired requests** for which the default and both single-detour policies failed, and the two-detour policy succeeded. The path and quantum draws were identical across these four runs.

The replay in results/reference/witness.json holds quantum readiness at 2.003 ms for all four plans. The default and individual detours reached frame readiness around 7.04–7.13 ms and gave fidelity around 0.835–0.837. The combined detours reached frame readiness at 3.118 ms and fidelity 0.919. This replay was selected after the run to illustrate the mechanism. Path B still won the overall reference comparison.

In the planner sensitivity run, each policy starts with an assumed 0.10 ms online selection delay and simulated activation packets. Adding 2 ms to joint routing alone reduces its 0.65-profile gain from +1.31 to −0.38 percentage points. The test does not time a deployed controller. Keeping the per-request decision path short is important to the next implementation.

## Protocol and physical model

The five-node topology offers quantum paths A–R1–R2–B and A–R3–B. At request arrival the controller uses one previously calibrated plan. The clock includes its assumed decision time and classical activation packet delivery. Link attempts start when both incident nodes are active. An intermediate node schedules a swap when its two local links are ready, even if a distant link is still generating. Each intermediate node sends its result directly to B over its chosen classical route. The application service requires B to receive all successful results to compute the Pauli frame. The runtime follows local swap readiness and uses a one-endpoint collector. QNP's published TRACK exchanges are a separate protocol implementation. [QNP paper](https://arxiv.org/pdf/2010.02575) · [RFC 9340](https://www.rfc-editor.org/rfc/rfc9340.html#section-4.4.2)

Each directed classical edge has a finite FIFO with packet serialization, propagation, and tail drop. Background packets are synthetic Poisson arrivals with alternating intensity. The mean offered load on R2→B is 0.25, 0.60, or 0.90. The mean on R3→B is 35% of that level. Outcomes are 96 bytes and activation messages are 64 bytes. The accounting unit is one byte transmitted over one link. Packet sizes, capacity, propagation and queue assumptions are listed in each manifest.

Quantum link success, herald timing and Bell-state measurement outcomes are synthetic. The elementary fidelity applies at heralding. The model tracks each quantum memory half from herald to its local swap or to endpoint frame availability. Conditional on successful operations, Werner visibilities multiply. For a live memory with visibility lifetime tau and age w, the visibility factor is exp(−w/tau). After the last swap, the two endpoint memories continue to age until the classical frame is available. The gate wait allowance is the time until this decay crosses the fidelity threshold. The audit independently recomputes delivered fidelity from the saved ages and checks the actual deadline. The model uses one request per traffic episode, so it does not measure memory contention among concurrent quantum requests or online adaptation.

### Compared policies

- **Default forwarding** chooses the quantum path using calibration outcomes with the default classical routes.
- **Sequential** fixes that quantum path first, then chooses its best permitted classical routes with the same joint deadline-and-fidelity criterion as the joint policy.
- **Mean joint** checks mean latency and fidelity among frame-complete calibration trials, then ranks by the frame-completion fraction. It is a simple expected-value comparator.
- **Joint** enumerates both quantum paths and all permitted classical plans. It maximizes the empirical probability of deadline-and-fidelity-qualified delivery on calibration trials.

All policies use the same event model, candidate paths, queue settings and known condition. Plans are frozen before holdout evaluation. In the exact reservation experiment, the sequential policy assigns quantum paths first using default-route calibration outcomes. It then optimizes classical routes within the exact remaining reservation. Joint allocation optimizes both decisions together by dynamic programming.

## Run the experiment

Use Python 3.10 or later. The supplied outputs were generated with Python 3.12.14. No third-party Python packages are required. From the extracted qcontrolflow_pilot directory, run

~~~sh
python3 -m unittest -v
python3 run_pilot.py --output rerun/reference
for probability in 0.55 0.65 0.75 0.85; do
  python3 run_pilot.py --seed 20261113 --secondary-success "$probability" --output "rerun/sweep_$probability"
done
python3 compare_budget_matched.py rerun/reference rerun/sweep_* --output rerun/exact_reservation_comparison.csv
python3 audit_pilot.py rerun
~~~

Each profile contains a manifest, calibration results, all held-out events, by-plan and by-policy summaries, the route-bundle comparison, an event trace, and the planner sensitivity sweep. The root exact_reservation_comparison.csv contains the full reservation grid. The independent audit checked all **108,000** raw holdout event rows and recomputed **845** exact reservation comparisons. Ten unit tests cover timing, memory decay, queue service, asynchronous operation, lost outcomes, control route complementarity, matched seeds, and exact optimization.

The next evidence needed for the manuscript is reproduction of the original results, followed by a comparison on the original and published protocol implementations with measured controller costs, calibrated queues and more independent topologies.
