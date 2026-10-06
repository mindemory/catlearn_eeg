#!/bin/bash
# Timed RAM, 3-block design: color fractals, all 18 spatial configs x 3 scenarios
# (A-A-B, A-A-AB, AB-AB-A) x 5 agents, from scratch, 3 blocks x 1000 updates x 32 trials.
#
# Three sequential GPU jobs of 6 configs (90 networks) each: ~38 min and ~7.4 GB peak
# GPU memory per job (measured on the M4 Pro, 24 GB). A failed job is reported and the
# next one still runs. Outputs: one folder per config,
#   ~/Documents/data/catlearn_eeg/ram_model/2x2/timed_config<NN>_graded_scratch_fractals-color/
# each with DATA_FORMAT.md describing every saved array.
#
# Launch detached, keeping the Mac awake, with a log:
#   nohup caffeinate -i bash runs/fractals_color_all_configs.sh > <log> 2>&1 &

cd "$(dirname "$0")/.." || exit 1
PY="$HOME/miniforge3/envs/kernelbehav/bin/python"
echo "=== started $(date) on $(hostname) ==="
for chunk in "1 2 3 4 5 6" "7 8 9 10 11 12" "13 14 15 16 17 18"; do
    echo "=== configs $chunk: $(date) ==="
    # shellcheck disable=SC2086  # word-splitting of $chunk is intended
    "$PY" R05_gpu_blocks.py --configs $chunk --agents 5 --updates 1000 --batch 32 \
        --stimuli fractals --color --sensor graded \
        || echo "!!! job for configs $chunk FAILED (exit $?)"
done
echo "=== all jobs finished $(date) ==="
