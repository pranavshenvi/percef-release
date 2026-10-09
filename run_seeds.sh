#!/usr/bin/env bash
# Seed repeats: the paper's numbers come from one random draw (seed 42) of the R/test query split,
# the 200 P calibration points and the KS sample. This repeats the default PercEF recipe, Ada-ef and
# the fixed-ef grid with 3 more seeds on 4 datasets (2 non-Gaussian, 2 near-Gaussian), settings P and
# R, on the same index. Not for LAION / Yambda (their queries come from the corpus).
#   tmux new -s seeds   then   bash run_seeds.sh
# Results: results_unified_<ds>_sweep_s<seed>_<ts>; own caches; the paper's runs are untouched.
# Summary: analysis/summarize_seeds.py. Log: seeds_<timestamp>.log.
set -u
cd "$(dirname "$0")"
LOG=seeds_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

DATASETS=${DATASETS:-"sift128 deepimage96 glove100 msturing"}
SEEDS=${SEEDS:-"1 2 3"}

if pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; then
  echo "[$(date)] another job is running; waiting for it to finish"
  while pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; do sleep 60; done
fi

echo "[$(date)] smoke test of --seed"
if ! python3 benchmark_unified.py --dataset sift128 --smoke --quick --seed 7 > seeds_smoke.log 2>&1; then
  echo "SMOKE TEST FAILED: not starting. Last lines of seeds_smoke.log:"; tail -n 30 seeds_smoke.log; exit 1
fi
echo "[$(date)] smoke test passed"

for s in $SEEDS; do
  for d in $DATASETS; do
    echo "[$(date)] $d seed $s"
    if python3 benchmark_unified.py --dataset "$d" --quick --seed "$s" 2>&1 | grep -A10 "SCORECARD"; then
      echo "[$(date)] $d seed $s done"
    else
      echo "[$(date)] $d seed $s FAILED (see the newest results_unified_${d}_sweep_s${s}_*/benchmark_unified.log)"
    fi
  done
  python3 analysis/summarize_seeds.py . | sed -n '/^Across seeds/,$p'
done
echo "[$(date)] done (log: $LOG). Copy back: results_unified_*_sweep_s*  seed_summary.csv  $LOG"
