# QControlFlow

Artifact for **QControlFlow: Fidelity-Constrained Co-Routing for Asynchronous Entanglement Delivery**. The study asks when classical outcome routing changes whether a quantum path can deliver a Bell pair that meets its fidelity threshold and deadline.

The artifact contains a controlled five-node pilot and a concurrent discrete-event campaign. The latter has **4,678 runs and 148,508 request–policy records** across six synthetic graphs and two graphs using Abilene and Polska adjacency. Graph reference data contain adjacency only. All queue, workload, link, controller and memory settings are modeled.

## Findings

- In the fixed-quantum-path pilot at load 0.90, minimum fidelity 0.85 and a 10 ms deadline, changing both outcome paths qualified 1,053 of 1,200 requests. The default and the two isolated changes qualified 571, 593 and 591. For 443 offers, only the combination qualified.
- In the concurrent campaign, direct joint selection qualified 55.21% of all offers compared with 57.48% for sequential selection. At equal 0.30 ms planning time the two policies qualified 57.66% and 57.89%.
- On fresh workload seeds, factorization reduced the median joint selection time from 0.524 to 0.199 ms. Optimized joint and optimized sequential selection qualified 57.13% and 57.27% of offers. The fixed-clock replay checks that factorization preserves simulated behavior.
- In the modeled shared-risk outage, the joint policy with one optional simultaneous outcome copy qualified 32.48% versus 27.35% for sequential and 31.20% for a delayed retry. All use the same traffic cap. Consult the reported actual traffic and admission rates.

See [paired comparisons](results/paired_effects.csv), [full summary](results/summary.json), and [validation](results/validation.json).

## Reproduce

Python 3.11 or newer with NumPy and Matplotlib is required. Published outputs were generated with Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.10.8.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python campaign.py --suite smoke --jobs 1 --fixed-planner-ms 0.30 --output smoke
bash reproduce.sh
python -m unittest discover -s pilot -p 'test_*.py'
python pilot/audit_pilot.py pilot/results
sha256sum -c SHA256SUMS
```

The full rerun writes to `reproduced/` by default. Set `QCF_WORKERS` to change the number of worker processes, or pass another output directory to `reproduce.sh`. Runtime-sensitive results vary with Python, processor, system load and process scheduling. Fixed-clock replays hold the simulated decision latency fixed at 0.30 ms. The exported `selection_cpu_ms` field is elapsed host wall time, including any scheduling interruptions.

| Directory | Purpose | Runs |
| --- | --- | ---: |
| `results/main` | Main measured-time comparisons | 1,120 |
| `results/equal_clock` | Same offered tapes with equal 0.30 ms decision latency | 1,120 |
| `results/confirm` | Fresh seeds for gated and factored scoring | 1,120 |
| `results/confirm_equal_clock` | Fresh seeds with equal 0.30 ms decision latency | 1,120 |
| `results/stress` | Shared-risk outage, alternative storage channel, and paired warm/cold runs | 198 |
| `pilot/results` | Five-node results, matched-budget comparisons and event witness | See pilot README |

Each campaign folder contains settings in `manifest.json`, one row per run in `runs.csv`, and the request-level trace in `requests.csv.gz`. `analyze.py` checks pairing, budget, terminal outcome classification and fixed-clock scorer equivalence before rebuilding summaries and figures. Request-level `reserved_byte_hops` includes activations, outcomes, and any planned optional copy and ACK. The run-level total also includes telemetry and counts serialized hops through drops. Physical random draws are indexed by request and component, so policies share exogenous draws where their choices use the same component.

- [network.py](network.py) generates the topology and background tape and executes finite FIFO queues.
- [simulator.py](simulator.py) executes activation, heralds, local swaps, frame collection, expiration and reservation accounting.
- [planner.py](planner.py) implements the direct and factored forecasts and the routing policies.
- [validate.py](validate.py) checks FIFO and the fidelity calculations against independent recurrences and density matrices.

## Scope

This is a standalone evaluation of single-collector frame-aware delivery. The chained comparison implements a one-way TRACK-inspired mechanism. The paper's earlier Figure 3 simulator and full published QNP implementation are separate reproduction tasks. The memory ledger, hardware settings and traffic are simulated. The synthetic outage is a stress case for correlated control loss.

The two reference adjacencies come from SNDlib via TopoHub. See [data provenance and attribution](data/README.md).
