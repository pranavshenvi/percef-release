#!/usr/bin/env python3
"""
Train LAET's predictor (Li et al., SIGMOD 2020) the way DARTH's authors implement it
(benchmarking-darth/notebooks_scripts/laet_training.py), on the training-trace log DARTH also
trains on. After F distance computations the search predicts how many distance computations the
query needs in total; LightGBM regressor, objective "regression", random_state=42.

Features, in the order the C++ predictor (LAETPredictorHNSW::predict_distance_calcs) feeds them:
  first_nn_dist, nn_dist, nn10_dist, nn_to_first, nn10_to_first, then the query's coordinates.
Target: the fewest distance computations at which the query reaches its highest recall in the
trace ("dists_for_max_recall"). The features come from the first logged row at or after F
distance computations (the trace is logged every LI computations; the C++ predicts at exactly F).

Usage: python3 laet_train.py train_log.csv train.fvecs F model.txt > info.json
"""
import sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

log, qfile, F, model_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32)


Q = read_fvecs(qfile)
cols = ["qid", "dists", "first_nn_dist", "nn_dist", "nn10_dist", "nn_to_first", "nn10_to_first", "r"]
df = pd.read_csv(log, usecols=cols)

target = df.loc[df.groupby("qid")["r"].transform("max") == df["r"]].groupby("qid")["dists"].min()
at_F = df[df["dists"] >= F].sort_values(["qid", "dists"]).groupby("qid").first()
qids = at_F.index.intersection(target.index)
qids = qids[qids < len(Q)]
X = np.hstack([at_F.loc[qids, ["first_nn_dist", "nn_dist", "nn10_dist", "nn_to_first", "nn10_to_first"]].to_numpy(),
               Q[qids.to_numpy()]]).astype(np.float64)
y = target.loc[qids].to_numpy(float)

model = lgb.LGBMRegressor(objective="regression", random_state=42, verbose=-1)
t0 = time.time()
model.fit(X, y)
train_s = time.time() - t0
model.booster_.save_model(model_path)
info = dict(F=F, queries=int(len(qids)), skipped=int(df["qid"].nunique() - len(qids)), train_s=round(train_s, 2),
            target_median=float(np.median(y)), model=model_path)
json.dump(info, sys.stdout, indent=1)
print(f"LAET F={F}: trained on {len(qids)} queries ({info['skipped']} ended before F) in {train_s:.1f}s; "
      f"median target {info['target_median']:.0f} distance computations", file=sys.stderr)
