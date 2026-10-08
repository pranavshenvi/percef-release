#!/usr/bin/env python3
"""
Choose DARTH's initial and minimum prediction intervals (ipi, mpi) on validation queries, with the
grid structure of the authors' experiments/hnsw_validation_interval_tuning.sh: ipi in D x {1, 1/2,
1/4, 1/8}, mpi in D x {1, 1/2, 1/4, 1/8, 1/10, 1/16, 1/20} with mpi <= ipi, where D is the mean
number of distance computations a training query needs to reach the target (darth_train.py).
Picks the pair with the fewest mean distance computations whose validation recall reaches the
target; if none does, the pair with the highest recall.

Usage: python3 darth_tune.py --bin hnsw_test --name CUSTOM_X --data dir/ --index x.index
         --model model.txt --efs 2000 --k 100 --nval 500 --target 0.95 --D 1234 --out tune_dir
"""
import os, sys, json, argparse, subprocess
import pandas as pd

ap = argparse.ArgumentParser()
for k in ("bin", "name", "data", "index", "model", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--efs", type=int, required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--nval", type=int, required=True)
ap.add_argument("--target", type=float, default=0.95)
ap.add_argument("--D", type=float, required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

ipis = sorted({max(1, round(a.D * f)) for f in (1, 1 / 2, 1 / 4, 1 / 8)}, reverse=True)
mpis = sorted({max(1, round(a.D * f)) for f in (1, 1 / 2, 1 / 4, 1 / 8, 1 / 10, 1 / 16, 1 / 20)}, reverse=True)
results = []
for ipi in ipis:
    for mpi in mpis:
        if mpi > ipi:
            continue
        log = os.path.join(a.out, f"val_ipi{ipi}_mpi{mpi}.csv")
        cmd = [a.bin, "--dataset", a.name, "--M", "16", "--efConstruction", "500", "--efSearch", str(a.efs),
               "--query-num", str(a.nval), "--k", str(a.k), "--mode", "early-stop-testing",
               "--index-filepath", a.index, "--dataset-dir-prefix", a.data, "--query-type", "validation",
               "--target-recall", str(a.target), "--initial-prediction-interval", str(ipi),
               "--min-prediction-interval", str(mpi), "--predictor-model-path", a.model, "--output", log]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        df = pd.read_csv(log).groupby("qid").last()
        r = dict(ipi=ipi, mpi=mpi, recall=float(df["r_actual"].mean()), dists=float(df["dists"].mean()))
        results.append(r)
        print(f"  ipi {ipi:>6} mpi {mpi:>6}: recall {r['recall']:.4f}, dists {r['dists']:.0f}", file=sys.stderr)

ok = [r for r in results if r["recall"] >= a.target]
best = min(ok, key=lambda r: r["dists"]) if ok else max(results, key=lambda r: r["recall"])
json.dump(dict(best=best, met_target=bool(ok), grid=results, D=a.D), sys.stdout, indent=1)
print(f"chosen ipi {best['ipi']} mpi {best['mpi']} (validation recall {best['recall']:.4f})", file=sys.stderr)
