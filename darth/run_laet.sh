#!/usr/bin/env bash
# LAET (Li et al., SIGMOD 2020) on a dataset that already has a DARTH run, through the LAET search
# in DARTH's own FAISS fork (--mode laet-early-stop-testing). Reuses that run's FAISS index, its
# training traces (train_log.csv, regenerated if it was deleted) and its FAISS fixed-ef grid, so
# LAET, DARTH and the fixed ef are scored on the same index and the same test queries.
#   bash darth/run_laet.sh <dataset> [efSearch upper bound, default 2000]
# Output: laet_test.csv, laet_tune.json and laet/ inside the newest results_darth_<dataset>_*,
# and an updated summary.json / printout with a LAET row. One thread, as every other method.
set -euo pipefail
DS=$1
EFS=${2:-2000}
TARGET=0.95
LI=${LI:-10}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BIN=${DARTH_ROOT:-$HOME/darth-build}/DARTH/build/hnsw-test/hnsw_test
export LD_LIBRARY_PATH="$HOME/lightgbm-install/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
NAME=CUSTOM_$(echo "$DS" | tr '[:lower:]' '[:upper:]')
DATA=$REPO/darth_data/
IDX=$REPO/darth_data/$NAME/faiss_M16_efC500.index
OUT=$(ls -d "$REPO"/results_darth_"${DS}"_2* 2>/dev/null | sort | tail -1 || true)
[ -n "$OUT" ] || { echo "no DARTH run for $DS: run bash darth/run_darth.sh $DS first"; exit 1; }
cd "$REPO"
exec > >(tee -a "$OUT/laet.log") 2>&1
read -r K NTRAIN NVAL NTEST < <(python3 -c "import json;m=json.load(open('$DATA/$NAME/meta.json'));print(m['k'],m['n_train'],m['n_val'],m['n_test'])")
echo "== LAET on $DS ($NAME), reusing $OUT: k=$K train=$NTRAIN val=$NVAL test=$NTEST"

run() { "$BIN" --dataset "$NAME" --M 16 --efConstruction 500 --k "$K" --index-filepath "$IDX" \
               --dataset-dir-prefix "$DATA" "$@"; }

T0=$(date +%s)
if [ ! -f "$OUT/train_log.csv" ]; then
  echo "== training traces were deleted: regenerating ($NTRAIN queries, every $LI distance computations)"
  run --efSearch "$EFS" --query-num "$NTRAIN" --mode early-stop-training --query-type training \
      --logging-interval "$LI" --output "$OUT/train_log.csv" > /dev/null
fi
D=$(python3 -c "import json;print(json.load(open('$OUT/train.json'))['D'])")
echo "== training and tuning LAET (F in D x {1/8,1/4,1/2}, D = $D; multiplier by bisection on $NVAL validation queries)"
python3 darth/laet_tune.py --bin "$BIN" --name "$NAME" --data "$DATA" --index "$IDX" \
    --log "$OUT/train_log.csv" --train-q "$DATA/$NAME/train.fvecs" --efs "$EFS" --k "$K" --nval "$NVAL" \
    --target "$TARGET" --D "$D" --li "$LI" --out "$OUT/laet" > "$OUT/laet_tune.json"
echo "laet_offline_s $(( $(date +%s) - T0 ))" >> "$OUT/offline_time.txt"
read -r F MULT < <(python3 -c "import json;b=json.load(open('$OUT/laet_tune.json'))['best'];print(b['F'],b['m'])")

echo "== LAET test: $NTEST queries, F=$F multiplier=$MULT"
run --efSearch "$EFS" --query-num "$NTEST" --mode laet-early-stop-testing --query-type testing \
    --target-recall "$TARGET" --fixed-amount-of-search "$F" --prediction-multiplier "$MULT" \
    --predictor-model-path "$OUT/laet/laet_model.txt" --output "$OUT/laet_test.csv" | { grep -E "Recall@" || true; }

OURS=$(ls -d "$REPO"/results_unified_"${DS}"_2* 2>/dev/null | sort | tail -1 || true)
python3 darth/darth_summarize.py "$OUT" ${OURS:+"$OURS"}
echo "== done: $OUT"
