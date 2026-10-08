#!/usr/bin/env bash
# Runs controlled-factor configs back to back, then summarizes.
#   bash run_controlled_experiments.sh          # every config
#   bash run_controlled_experiments.sh A        # only Experiment A (SIFT, no synthetic data needed)
#   bash run_controlled_experiments.sh BC       # only the anisotropy-based synthetic experiments
#   bash run_controlled_experiments.sh D        # only the cluster-separation synthetic experiment
# Config names come from controlled_datasets.CONFIGS, so they always match the
# generator settings. Configs whose caches exist resume quickly.
set -u
EXPS="${1:-ABCD}"

python3 controlled_datasets.py --only "$EXPS" || exit 1

CONFIGS=$(python3 -c "from controlled_datasets import CONFIGS; print(' '.join(n for n, c in CONFIGS.items() if c[3] in '$EXPS'))")
for cfg in $CONFIGS; do
  echo "=== $(date '+%F %T') starting $cfg"
  python3 -u benchmark_controlled.py --config "$cfg" || echo "!!! $cfg failed, continuing"
done

python3 summarize_controlled.py
