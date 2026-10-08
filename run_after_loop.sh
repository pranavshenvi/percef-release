#!/usr/bin/env bash
# Wait for the running benchmark loop to finish, then run the probe-length ablation and the DARTH
# comparison, one job at a time (so latency is measured on an otherwise idle machine).
#   tmux new -s after   then   bash run_after_loop.sh
# Needs, beforehand (once): sudo apt-get install -y cmake g++ libopenblas-dev libomp-dev
#                           pip install --user lightgbm pandas
# Everything is logged to after_loop_<timestamp>.log. A failing dataset is reported and skipped.
set -u
cd "$(dirname "$0")"
LOG=after_loop_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

ABLATION=${ABLATION:-"sift128 deep1b deepimage96 bigann glove100 msturing"}
DARTH=${DARTH:-"glove100 sift128 deep1b msturing"}

echo "[$(date)] waiting for the current benchmark_unified.py runs to finish"
idle=0
while [ "$idle" -lt 3 ]; do                 # 3 consecutive minutes with no benchmark process
  if pgrep -f "benchmark_unified.py" > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  sleep 60
done
echo "[$(date)] loop finished"

for d in $ABLATION; do
  echo "[$(date)] ablation: $d"
  if python3 benchmark_unified.py --dataset "$d" --ablation --quick --settings R 2>&1 | grep -A12 "SCORECARD"; then
    echo "[$(date)] ablation $d done"
  else
    echo "[$(date)] ablation $d FAILED (see results_unified_${d}_ablation_*/benchmark_unified.log)"
  fi
done

echo "[$(date)] building DARTH"
if ! bash darth/setup_darth.sh > darth_setup.log 2>&1; then
  echo "[$(date)] DARTH build FAILED: see darth_setup.log (last lines below); skipping the DARTH runs"
  tail -n 30 darth_setup.log
  exit 1
fi
for d in $DARTH; do
  echo "[$(date)] DARTH: $d"
  bash darth/run_darth.sh "$d" || echo "[$(date)] DARTH $d FAILED (see results_darth_${d}_*/run.log)"
done
echo "[$(date)] all done. Results: results_unified_*_ablation_*  results_darth_*  (log: $LOG)"
