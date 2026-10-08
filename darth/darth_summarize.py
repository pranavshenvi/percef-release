#!/usr/bin/env python3
"""
Summarise one DARTH run (results_darth_<dataset>_<ts>/) with the same scorecard as
benchmark_unified.py: mean recall, p1/p5, target-hit share, distance computations and latency per
query, and the saving against a tuned fixed ef *of the same library* (FAISS HNSW here, HNSWlib for
ours and Ada-ef) at equal mean recall, log(1 - recall) vs log(cost) interpolation. Latency is
compared only within a library: FAISS times each query in C++, our harness through Python.

Usage: python3 darth_summarize.py results_darth_<dataset>_<ts> [results_unified_<dataset>_<ts>]
"""
import os, re, sys, glob, json
import numpy as np
import pandas as pd

TARGET = 0.95


def per_query(path):
    df = pd.read_csv(path).groupby("qid").last()          # final row of each query
    r = (df["r_actual"] if "r_actual" in df else df["r"]).to_numpy(float)
    return r, df["dists"].to_numpy(float), df["elaps_ms"].to_numpy(float) * 1000


def row(name, r, dc, lat, **extra):
    return dict(name=name, mean_r=float(r.mean()), p1=float(np.percentile(r, 1)), p5=float(np.percentile(r, 5)),
                pct_target=float(np.mean(r >= TARGET) * 100), total_dc=float(dc.mean()),
                mean_lat_us=float(lat.mean()), **extra)


def fixed_interp(fixed, goal, key):
    x = lambda v: np.log(max(1.0 - v, 1e-6))
    for a, b in zip(fixed, fixed[1:]):
        if a["mean_r"] < goal <= b["mean_r"]:
            t = (x(goal) - x(a["mean_r"])) / (x(b["mean_r"]) - x(a["mean_r"]))
            if a.get(key) is None or b.get(key) is None:   # e.g. no latency in runs before Oct 2026
                return None
            if key in ("p1", "p5"):
                return a[key] + t * (b[key] - a[key])
            return float(np.exp(np.log(a[key]) + t * np.log(b[key] / a[key])))
    return None


def vs_fixed(r, fixed):
    out = {}
    for key, nm in (("total_dc", "saving_dc_pct"), ("mean_lat_us", "saving_lat_pct")):
        f = fixed_interp(fixed, r["mean_r"], key)
        out[nm] = None if f is None else (f - r[key]) / f * 100
    for key in ("p1", "p5"):
        f = fixed_interp(fixed, r["mean_r"], key)
        out[key + "_gain"] = None if f is None else r[key] - f
    return out


run = sys.argv[1]
fixed = []
for p in glob.glob(os.path.join(run, "fixed_ef*.csv")):
    ef = int(re.search(r"fixed_ef(\d+)\.csv", p).group(1))
    fixed.append(row(f"FAISS fixed(ef={ef})", *per_query(p), avg_ef=ef))
fixed.sort(key=lambda r: r["avg_ef"])
darth = row("DARTH", *per_query(os.path.join(run, "darth_test.csv")))
darth["scorecard"] = vs_fixed(darth, fixed)
laet = None                                              # darth/run_laet.sh adds LAET to the same run
if os.path.exists(os.path.join(run, "laet_test.csv")):
    laet = row("LAET", *per_query(os.path.join(run, "laet_test.csv")))
    laet["scorecard"] = vs_fixed(laet, fixed)

fp = lambda v, f="{:+.1f}%": "n/a" if v is None else f.format(v)
print(f"\n  {'method':<34} {'meanR':>7} {'p1':>6} {'hit%':>6} {'DC':>8} {'lat us':>8} {'save DC':>8} {'save lat':>9} {'p1 gain':>8}")
for f in fixed:
    print(f"  {f['name']:<34} {f['mean_r']:>7.4f} {f['p1']:>6.3f} {f['pct_target']:>6.1f} {f['total_dc']:>8.0f} {f['mean_lat_us']:>8.0f}")
sc = darth["scorecard"]
print(f"  {'DARTH (FAISS)':<34} {darth['mean_r']:>7.4f} {darth['p1']:>6.3f} {darth['pct_target']:>6.1f} "
      f"{darth['total_dc']:>8.0f} {darth['mean_lat_us']:>8.0f} {fp(sc['saving_dc_pct']):>8} {fp(sc['saving_lat_pct']):>9} "
      f"{fp(sc['p1_gain'], '{:+.3f}'):>8}")
if laet:
    sc = laet["scorecard"]
    print(f"  {'LAET (FAISS)':<34} {laet['mean_r']:>7.4f} {laet['p1']:>6.3f} {laet['pct_target']:>6.1f} "
          f"{laet['total_dc']:>8.0f} {laet['mean_lat_us']:>8.0f} {fp(sc['saving_dc_pct']):>8} {fp(sc['saving_lat_pct']):>9} "
          f"{fp(sc['p1_gain'], '{:+.3f}'):>8}")

ours = None
if len(sys.argv) > 2:                                   # our HNSWlib run on the same dataset, setting R
    # Scored here from rows_R.json, which every run has (summaries before Oct 2026 lack the scorecard).
    with open(os.path.join(sys.argv[2], "rows_R.json")) as f:
        rows = {r["name"]: r for r in json.loads(f.read().replace("NaN", "null"))}
    hfixed = sorted((r for r in rows.values() if r["name"].startswith("Fixed")), key=lambda r: r["avg_ef"])
    ours = {}
    print(f"\n  Same queries, HNSWlib ({os.path.basename(sys.argv[2].rstrip('/'))}), setting R, "
          f"each against its own library's fixed ef:")
    for n in ("Ada-ef (as shipped)", "Ada-ef (WAE floor)", "Ours (K=1, Isotonic)"):
        if n not in rows:
            continue
        r, c = rows[n], vs_fixed(rows[n], hfixed)
        ours[n] = dict(mean_r=r["mean_r"], p1=r["p1"], pct_target=r["pct_target"], **c)
        lat = "n/a" if r.get("mean_lat_us") is None else f"{r['mean_lat_us']:.0f}"
        print(f"  {n:<34} {r['mean_r']:>7.4f} {r['p1']:>6.3f} {r['pct_target']:>6.1f} {r['total_dc']:>8.0f} "
              f"{lat:>8} {fp(c['saving_dc_pct']):>8} {fp(c['saving_lat_pct']):>9} {fp(c['p1_gain'], '{:+.3f}'):>8}")

with open(os.path.join(run, "summary.json"), "w") as f:
    json.dump(dict(darth=darth, laet=laet, faiss_fixed=fixed, hnswlib_setting_R=ours), f, indent=1)
print(f"\n  wrote {run}/summary.json")
