#!/usr/bin/env bash
# Robustness sweep for the paper: do the comparisons hold at other target recalls and k?
#   target recall 0.90 and 0.99 (k = dataset default, 100), and k = 10 (target 0.95)
#   on 4 non-Gaussian and 2 near-Gaussian datasets, settings P and R, PercEF's default recipe only.
# The paper's own runs (0.95, k = 100) are the reference point and are not re-run.
#   tmux new -s sweep   then   bash run_sweep.sh
# One job at a time (latency is measured), operating point by operating point, so a partial run is
# already usable. Results: results_unified_<ds>_sweep_t<target>[_k<k>]_<ts>; separate caches, so the
# paper's runs and caches are untouched. Summary at the end: analysis/summarize_sweep.py.
# Log: sweep_<timestamp>.log. A failing run is reported and skipped.
set -u
cd "$(dirname "$0")"
LOG=sweep_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

DATASETS=${DATASETS:-"deepimage96 deep1b sift128 vibe_landmark_dino glove100 msturing"}
POINTS=${POINTS:-"--target-recall=0.90 --k=10 --target-recall=0.99"}

if pgrep -f "benchmark_unified.py" > /dev/null; then
  echo "[$(date)] another benchmark is running; waiting for it to finish"
  while pgrep -f "benchmark_unified.py" > /dev/null; do sleep 60; done
fi

echo "[$(date)] smoke test of the new flags (tiny query sets, separate smoke cache)"
for p in --k=10 --target-recall=0.99; do
  if ! python3 benchmark_unified.py --dataset sift128 --smoke --quick "$p" > sweep_smoke.log 2>&1; then
    echo "SMOKE TEST FAILED for $p: not starting the sweep. Last lines of sweep_smoke.log:"
    tail -n 30 sweep_smoke.log
    exit 1
  fi
done
echo "[$(date)] smoke test passed"

for p in $POINTS; do
  for d in $DATASETS; do
    echo "[$(date)] $d $p"
    if python3 benchmark_unified.py --dataset "$d" --quick "$p" 2>&1 | grep -A10 "SCORECARD"; then
      echo "[$(date)] $d $p done"
    else
      echo "[$(date)] $d $p FAILED (see the newest results_unified_${d}_sweep_*/benchmark_unified.log)"
    fi
  done
  python3 analysis/summarize_sweep.py . | sed -n '/^Counts per operating point/,$p'
done
echo "[$(date)] sweep done (log: $LOG). Copy back: results_unified_*_sweep_*  sweep_summary.csv  sweep_counts.csv  $LOG"
