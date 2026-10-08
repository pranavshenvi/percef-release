#!/usr/bin/env bash
# Learned baselines for the paper / revision: LAET on the 4 datasets DARTH already ran on, then
# DARTH + LAET on 3 more (one Ada-ef dataset, one more non-Gaussian learned embedding, one more
# near-Gaussian text embedding). One job at a time; a failing dataset is reported and skipped.
#   tmux new -s baselines   then   bash run_baselines.sh
# Log: baselines_<timestamp>.log. Results: inside results_darth_<dataset>_* (summary.json has
# DARTH, LAET, the FAISS fixed-ef grid and our methods on the same test queries).
set -u
cd "$(dirname "$0")"
LOG=baselines_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

LAET_ONLY=${LAET_ONLY:-"glove100 sift128 deep1b msturing"}
NEW=${NEW:-"deepimage96 vibe_landmark_dino dbpedia1536"}

if pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; then
  echo "[$(date)] another job is running; waiting for it to finish"
  while pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; do sleep 60; done
fi

for d in $LAET_ONLY; do
  echo "[$(date)] LAET: $d"
  bash darth/run_laet.sh "$d" || echo "[$(date)] LAET $d FAILED (see results_darth_${d}_*/laet.log)"
done
for d in $NEW; do
  echo "[$(date)] DARTH: $d"
  if bash darth/run_darth.sh "$d"; then
    echo "[$(date)] LAET: $d"
    bash darth/run_laet.sh "$d" || echo "[$(date)] LAET $d FAILED (see results_darth_${d}_*/laet.log)"
  else
    echo "[$(date)] DARTH $d FAILED (see results_darth_${d}_*/run.log)"
  fi
done
echo "[$(date)] all done (log: $LOG). Copy back the results_darth_* folders WITHOUT train_log.csv:"
echo "  tar --exclude=train_log.csv -czf baselines_results.tgz results_darth_* $LOG"
