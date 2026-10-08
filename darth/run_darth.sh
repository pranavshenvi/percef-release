#!/usr/bin/env bash
# DARTH on one of our datasets, same queries and ground truth as the other methods.
#   bash darth/run_darth.sh <dataset> [efSearch upper bound, default 2000]
# Steps: export -> FAISS HNSW index (M=16, efConstruction=500) -> training traces -> LightGBM
# predictor -> tune intervals on validation queries -> test at target recall 0.95 -> FAISS fixed-ef
# grid on the same test queries -> summary. Searches run on one thread (OMP_NUM_THREADS=1), as ours.
# Run from the repo root after `bash darth/setup_darth.sh`. Needs the dataset's benchmark caches.
set -euo pipefail
DS=$1
EFS=${2:-2000}
TARGET=0.95
LI=${LI:-10}                                # training-trace logging interval (distance computations)
REPO=$(cd "$(dirname "$0")/.." && pwd)
BIN=${DARTH_ROOT:-$HOME/darth-build}/DARTH/build/hnsw-test/hnsw_test
export LD_LIBRARY_PATH="$HOME/lightgbm-install/lib:${LD_LIBRARY_PATH:-}"
NAME=CUSTOM_$(echo "$DS" | tr '[:lower:]' '[:upper:]')
DATA=$REPO/darth_data/
OUT=$REPO/results_darth_${DS}_$(date +%Y%m%d_%H%M%S)
IDX=$REPO/darth_data/$NAME/faiss_M16_efC500.index
mkdir -p "$OUT"
cd "$REPO"
exec > >(tee -a "$OUT/run.log") 2>&1

[ -x "$BIN" ] || { echo "DARTH not built: run bash darth/setup_darth.sh"; exit 1; }
[ -f "$DATA/$NAME/meta.json" ] || python3 benchmark_unified.py --dataset "$DS" --export-darth darth_data
read -r K NTRAIN NVAL NTEST < <(python3 -c "import json;m=json.load(open('$DATA/$NAME/meta.json'));print(m['k'],m['n_train'],m['n_val'],m['n_test'])")
echo "== $DS ($NAME): k=$K train=$NTRAIN val=$NVAL test=$NTEST efSearch<=$EFS target=$TARGET"

run() { "$BIN" --dataset "$NAME" --M 16 --efConstruction 500 --k "$K" --index-filepath "$IDX" \
               --dataset-dir-prefix "$DATA" "$@"; }

if [ ! -f "$IDX" ]; then
  echo "== building FAISS HNSW index"
  ( export OMP_NUM_THREADS=$(nproc); "$BIN" --dataset "$NAME" --M 16 --efConstruction 500 \
      --efSearch "$EFS" --k "$K" --query-num 1 --mode no-early-stop --query-type testing \
      --index-filepath "$IDX" --dataset-dir-prefix "$DATA" --save-index ) | { grep -E "Index" || true; }
fi
export OMP_NUM_THREADS=1

echo "== training traces ($NTRAIN queries, every $LI distance computations)"
T0=$(date +%s)
run --efSearch "$EFS" --query-num "$NTRAIN" --mode early-stop-training --query-type training \
    --logging-interval "$LI" --output "$OUT/train_log.csv" > /dev/null
echo "== training the predictor"
python3 darth/darth_train.py "$OUT/train_log.csv" "$OUT/model.txt" --target "$TARGET" > "$OUT/train.json"
D=$(python3 -c "import json;print(json.load(open('$OUT/train.json'))['D'])")
echo "== tuning prediction intervals on $NVAL validation queries (D = $D)"
python3 darth/darth_tune.py --bin "$BIN" --name "$NAME" --data "$DATA" --index "$IDX" --model "$OUT/model.txt" \
    --efs "$EFS" --k "$K" --nval "$NVAL" --target "$TARGET" --D "$D" --out "$OUT/tune" > "$OUT/tune.json"
echo "offline_s $(( $(date +%s) - T0 ))" > "$OUT/offline_time.txt"
read -r IPI MPI < <(python3 -c "import json;b=json.load(open('$OUT/tune.json'))['best'];print(b['ipi'],b['mpi'])")

echo "== DARTH test: $NTEST queries, ipi=$IPI mpi=$MPI"
run --efSearch "$EFS" --query-num "$NTEST" --mode early-stop-testing --query-type testing \
    --target-recall "$TARGET" --initial-prediction-interval "$IPI" --min-prediction-interval "$MPI" \
    --predictor-model-path "$OUT/model.txt" --output "$OUT/darth_test.csv" | { grep -E "Recall@" || true; }

echo "== FAISS fixed-ef grid on the same test queries"
for EF in 100 125 150 175 200 250 300 350 400 500 600 700 800 1000 1250 1500 2000 3000 5000; do
  [ "$EF" -le "$EFS" ] || break
  run --efSearch "$EF" --query-num "$NTEST" --mode no-early-stop --query-type testing \
      --output "$OUT/fixed_ef$EF.csv" | { grep -E "Recall@" || true; } | sed "s/^/  ef=$EF /"
done

OURS=$(ls -d "$REPO"/results_unified_"${DS}"_2* 2>/dev/null | sort | tail -1 || true)
python3 darth/darth_summarize.py "$OUT" ${OURS:+"$OURS"}
echo "== done: $OUT"
