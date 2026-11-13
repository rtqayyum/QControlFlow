#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
qcf_python=${QCF_PYTHON:-python3}
qcf_workers=${QCF_WORKERS:-4}
qcf_output=${1:-reproduced}
mkdir -p "$qcf_output"

"$qcf_python" campaign.py --suite main --jobs "$qcf_workers" --seeds 4 --requests 32 --output "$qcf_output/main"
"$qcf_python" campaign.py --suite main --jobs "$qcf_workers" --seeds 4 --requests 32 --fixed-planner-ms 0.30 --output "$qcf_output/equal_clock"
"$qcf_python" campaign.py --suite stress --jobs "$qcf_workers" --seeds 3 --requests 26 --output "$qcf_output/stress"
"$qcf_python" campaign.py --suite confirm --jobs "$qcf_workers" --seeds 4 --requests 32 --output "$qcf_output/confirm"
"$qcf_python" campaign.py --suite confirm --jobs "$qcf_workers" --seeds 4 --requests 32 --fixed-planner-ms 0.30 --output "$qcf_output/confirm_equal_clock"
"$qcf_python" validate.py --output "$qcf_output/validation.json"
"$qcf_python" analyze.py --results "$qcf_output" --figures "$qcf_output/figures"
