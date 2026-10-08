#!/usr/bin/env bash
# Latency fix of 2026-10-06 (running probe score, no heap copy; updateAsOf061026.md §9).
# 1. rebuild the C++ extension, 2. check the fix picks the same ef as the calibration path,
# 3. re-run the 8 non-Gaussian datasets (settings P and R), one at a time on an idle machine.
#   tmux new -s retime   then   bash run_retime_probe_score.sh
# Recall and distance counts should not move (beyond a rare query whose score sits exactly on a
# rounding boundary); only latency should. Log: retime_<timestamp>.log
set -u
cd "$(dirname "$0")"
LOG=retime_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

DATASETS=${DATASETS:-"deepimage96 deep1b sift128 gist960 fashionmnist784 yambda vibe_landmark_dino vibe_inaturalist_resnet"}

echo "[$(date)] rebuilding the extension"
(cd chao_hybrid_ada_ef && rm -rf build chao_hybrid_ada_ef_cpp*.so && python3 setup.py build_ext --inplace > ../retime_build.log 2>&1) \
  || { echo "BUILD FAILED, see retime_build.log"; tail -n 30 retime_build.log; exit 1; }

echo "[$(date)] equivalence check"
PYTHONPATH=chao_hybrid_ada_ef python3 analysis/test_probe_score.py || { echo "CHECK FAILED: not running the benchmark"; exit 1; }

for d in $DATASETS; do
  echo "[$(date)] benchmark: $d"
  if python3 benchmark_unified.py --dataset "$d" 2>&1 | grep -A14 "SCORECARD"; then
    echo "[$(date)] $d done"
  else
    echo "[$(date)] $d FAILED (see the newest results_unified_${d}_*/benchmark_unified.log)"
  fi
done
echo "[$(date)] all done; now: python3 analysis/make_paper_figures.py   (log: $LOG)"
