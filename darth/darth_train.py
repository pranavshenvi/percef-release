#!/usr/bin/env python3
"""
Train DARTH's recall predictor on a training-trace log, exactly as the authors' own
notebooks_scripts/predictor_training.py does: LightGBM regressor, objective "regression",
n_estimators=100, random_state=42, on their "all_feats" feature set, target = recall so far.
Also reports D, the mean number of distance computations a training query needs to reach the
target recall; darth_tune.py builds DARTH's prediction-interval grid from it.

Usage: python3 darth_train.py train_log.csv model.txt --target 0.95 > train.json
"""
import sys, json, time, argparse
import pandas as pd
import lightgbm as lgb

# DARTH's "all_feats" = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats
FEATS = ["step", "dists", "inserts",
         "first_nn_dist", "nn_dist", "furthest_dist",
         "avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("model")
ap.add_argument("--target", type=float, default=0.95)
a = ap.parse_args()

df = pd.read_csv(a.log, usecols=["qid"] + FEATS + ["r"])
model = lgb.LGBMRegressor(objective="regression", random_state=42, n_estimators=100, verbose=-1)
t0 = time.time()
model.fit(df[FEATS], df["r"])
train_s = time.time() - t0
model.booster_.save_model(a.model)

reached = df[df["r"] >= a.target].groupby("qid")["dists"].min()
last = df.groupby("qid")["dists"].max()
need = reached.reindex(last.index).fillna(last)          # never reached: the whole search
out = dict(rows=int(len(df)), queries=int(df["qid"].nunique()), train_s=round(train_s, 2),
           D=float(need.mean()), reached_share=float(reached.size / last.size), model=a.model)
json.dump(out, sys.stdout, indent=1)
print(file=sys.stderr)
print(f"trained on {out['rows']} rows / {out['queries']} queries in {train_s:.1f}s; D = {out['D']:.0f}", file=sys.stderr)
