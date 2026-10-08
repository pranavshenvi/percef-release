#!/usr/bin/env bash
# The two candidate fixes from the robustness sweep (updateAsOf071026.md §5-6), on the sweep's datasets
# and operating points:
#   scaled : PercEF's ef table times one factor, chosen on the calibration queries to meet the target
#            mean recall (PercEF missed the target by a few points at 0.90 and on GloVe / MS Turing at 0.99)
#   L=30   : a 30-distance probe instead of 100, with and without scaling (at k = 10 PercEF saved no work)
#   tmux new -s fixes   then   bash run_fixes.sh
# Reuses the sweep's caches (true min ef, Ada-ef tables), so it mostly re-runs the test queries.
# Results: results_unified_<ds>_sweep[_t<target>][_k<k>]_fixes_<ts>; the paper's runs are untouched.
# Log: fixes_<timestamp>.log. Summary: analysis/summarize_sweep.py (reads the fixes folders too).
set -u
cd "$(dirname "$0")"
LOG=fixes_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

DATASETS=${DATASETS:-"deepimage96 deep1b sift128 vibe_landmark_dino glove100 msturing"}
# "base" = the paper's own operating point (target 0.95, dataset k)
POINTS=${POINTS:-"--k=10 base --target-recall=0.90 --target-recall=0.99"}

if pgrep -f "benchmark_unified.py" > /dev/null; then
  echo "[$(date)] another benchmark is running; waiting for it to finish"
  while pgrep -f "benchmark_unified.py" > /dev/null; do sleep 60; done
fi

echo "[$(date)] smoke test of --fixes"
if ! python3 benchmark_unified.py --dataset sift128 --smoke --quick --fixes --k=10 > fixes_smoke.log 2>&1; then
  echo "SMOKE TEST FAILED: not starting. Last lines of fixes_smoke.log:"
  tail -n 30 fixes_smoke.log
  exit 1
fi
grep -E "scaled|L=30" fixes_smoke.log | head -6
echo "[$(date)] smoke test passed"

for p in $POINTS; do
  extra=""; [ "$p" != "base" ] && extra="$p"
  for d in $DATASETS; do
    echo "[$(date)] $d $p"
    if python3 benchmark_unified.py --dataset "$d" --quick --fixes $extra 2>&1 | grep -E "scaled|L=30|SCORECARD"; then
      echo "[$(date)] $d $p done"
    else
      echo "[$(date)] $d $p FAILED (see the newest results_unified_${d}_sweep*_fixes_*/benchmark_unified.log)"
    fi
  done
  python3 analysis/summarize_sweep.py . | sed -n '/^Counts per operating point/,$p'
done
echo "[$(date)] done (log: $LOG). Copy back: results_unified_*_fixes_*  sweep_summary.csv  sweep_counts.csv  $LOG"
