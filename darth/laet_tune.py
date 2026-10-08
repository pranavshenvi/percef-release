#!/usr/bin/env python3
"""
Choose LAET's two settings on the validation queries (the same ones DARTH tunes on):
  F  -- distance computations before the prediction: D x {1/8, 1/4, 1/2}, rounded to the trace's
        logging interval, where D is the mean number a training query needs to reach the target
        (darth_train.py); one model per F (laet_train.py)
  m  -- the multiplier on the prediction (LAET's knob for the recall target): for each F, the
        smallest m in [0.5, 8] whose validation mean recall reaches the target (geometric bisection)
Picks the (F, m) with the fewest mean distance computations; if no m reaches the target, the
highest recall. Writes the chosen model to <out>/laet_model.txt.

Usage: python3 laet_tune.py --bin hnsw_test --name CUSTOM_X --data dir/ --index x.index
         --log train_log.csv --train-q train.fvecs --efs 2000 --k 100 --nval 500 --target 0.95
         --D 1234 --li 10 --out laet_dir > laet_tune.json
"""
import os, sys, json, shutil, argparse, subprocess
import pandas as pd

ap = argparse.ArgumentParser()
for k in ("bin", "name", "data", "index", "log", "train-q", "out"):
    ap.add_argument("--" + k, required=True)
for k in ("efs", "k", "nval", "li"):
    ap.add_argument("--" + k, type=int, required=True)
ap.add_argument("--target", type=float, default=0.95)
ap.add_argument("--D", type=float, required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))


def validate(F, m, model):
    log = os.path.join(a.out, f"val_F{F}_m{m:.4f}.csv")
    cmd = [a.bin, "--dataset", a.name, "--M", "16", "--efConstruction", "500", "--efSearch", str(a.efs),
           "--query-num", str(a.nval), "--k", str(a.k), "--mode", "laet-early-stop-testing",
           "--index-filepath", a.index, "--dataset-dir-prefix", a.data, "--query-type", "validation",
           "--target-recall", str(a.target), "--fixed-amount-of-search", str(F), "--prediction-multiplier", str(m),
           "--predictor-model-path", model, "--output", log]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    df = pd.read_csv(log).groupby("qid").last()
    return dict(F=F, m=m, recall=float(df["r"].mean()), dists=float(df["dists"].mean()))


results, best_per_F = [], []
for frac in (1 / 8, 1 / 4, 1 / 2):
    F = max(a.li, int(round(a.D * frac / a.li)) * a.li)
    model = os.path.join(a.out, f"laet_model_F{F}.txt")
    subprocess.run([sys.executable, os.path.join(here, "laet_train.py"), a.log, a.train_q, str(F), model],
                   check=True, stdout=open(os.path.join(a.out, f"train_F{F}.json"), "w"))
    lo, hi = 0.5, 8.0
    r_lo, r_hi = validate(F, lo, model), validate(F, hi, model)
    results += [r_lo, r_hi]
    if r_lo["recall"] >= a.target:
        best = r_lo
    else:
        best = r_hi
        if r_hi["recall"] >= a.target:
            for _ in range(7):                       # ratio 16 -> about 4% resolution on m
                mid = (lo * hi) ** 0.5
                r = validate(F, mid, model)
                results.append(r)
                if r["recall"] >= a.target:
                    hi, best = mid, r
                else:
                    lo = mid
    best_per_F.append(dict(best, model=model))
    print(f"  F {F:>6}: m {best['m']:.3f} -> validation recall {best['recall']:.4f}, dists {best['dists']:.0f}",
          file=sys.stderr)

ok = [b for b in best_per_F if b["recall"] >= a.target]
best = min(ok, key=lambda b: b["dists"]) if ok else max(best_per_F, key=lambda b: b["recall"])
shutil.copy(best["model"], os.path.join(a.out, "laet_model.txt"))
json.dump(dict(best=best, met_target=bool(ok), per_F=best_per_F, grid=results, D=a.D), sys.stdout, indent=1)
print(f"chosen F {best['F']} m {best['m']:.3f} (validation recall {best['recall']:.4f})", file=sys.stderr)
