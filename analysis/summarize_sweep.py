"""Robustness sweep (run_sweep.sh): do the paper's comparisons hold at other target recalls and k?

For every dataset and every operating point (target recall, k) -- the paper's own runs at 0.95 and
the dataset's default k included -- each method is scored against the tuned fixed ef, exactly as
make_paper_figures.py does: at the same mean recall (work, time, p1) and at the same p1 (time).
Prints a per-run table and, per operating point, the counts the paper quotes; writes sweep_summary.csv.

    python analysis/summarize_sweep.py server_results
"""
import csv, glob, json, os, re, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_paper_figures import METHODS, KS, BAND_HI, fixed_at, fixed_at_p1   # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "."
PAT = re.compile(r"results_unified_(.+?)(?:_sweep(?:_t([0-9.]+))?(?:_k(\d+))?(_fixes)?)?_(\d{8}_\d{6})$")
# --fixes runs (run_fixes.sh): only their extra PercEF variants are read; base methods come from the regular runs

# latest folder per (dataset, target, k); sweep datasets only
runs, sweep_ds = {}, set()
for d in sorted(glob.glob(os.path.join(root, "results_unified_*"))):
    b = os.path.basename(d)
    if "smoke" in b or "ablation" in b or re.search(r"_sweep_s\d", b) or not os.path.exists(os.path.join(d, "meta.json")):
        continue                                   # seed repeats: analysis/summarize_seeds.py
    m = PAT.match(b)
    if not m:
        continue
    meta = json.load(open(os.path.join(d, "meta.json")))
    ds = m.group(1)
    if "_sweep" in b:
        sweep_ds.add(ds)
    runs[(ds, round(float(meta["target_recall"]), 4), int(meta["K"]), bool(m.group(4)))] = d

out = []
for (ds, tgt, k, fixes), d in sorted(runs.items()):
    if ds not in sweep_ds:
        continue
    for S in ("P", "R"):
        f = os.path.join(d, f"rows_{S}.json")
        if not os.path.exists(f):
            continue
        rows = json.load(open(f))
        fixed = sorted([r for r in rows if r["name"].startswith("Fixed")], key=lambda r: r["avg_ef"])
        names = dict(METHODS)
        if fixes:
            names = {n: ("Ours " + n.split("Isotonic, ")[1].rstrip(")").replace("=", "").replace(", ", " "))
                     for n in (x["name"] for x in rows) if n.startswith("Ours (K=1, Isotonic, ")}
        for full, meth in names.items():
            r = next((x for x in rows if x["name"] == full), None)
            if r is None:
                continue
            fa = lambda key: fixed_at(fixed, r["mean_r"], key)
            sv = lambda key: None if fa(key) is None or r.get(key) is None else (fa(key) - r[key]) / fa(key) * 100
            f1 = lambda key: fixed_at_p1(fixed, r["p1"], key)
            sv1 = lambda key: None if f1(key) is None or r.get(key) is None else (f1(key) - r[key]) / f1(key) * 100
            out.append(dict(dataset=ds, ks=KS.get(ds), target=tgt, k=k, setting=S, method=meth,
                            mean_r=r["mean_r"], p1=r["p1"], scale=r.get("scale"),
                            p5_gain=None if fa("p5") is None else r["p5"] - fa("p5"),
                            p1_gain=None if fa("p1") is None else r["p1"] - fa("p1"),
                            save_dc=sv("total_dc"), save_lat=sv("mean_lat_us"),
                            save_dc_p1=sv1("total_dc"), save_lat_p1=sv1("mean_lat_us")))

if not out:
    sys.exit("no sweep runs found (results_unified_<dataset>_sweep_*); run run_sweep.sh first")

fm = lambda v, w=6: " " * (w - 3) + "n/a" if v is None else f"{v:+{w}.1f}"
print(f"{'dataset':24} {'tgt':>5} {'k':>5} S {'method':10} {'recall':>6} {'DC%':>6} {'time%':>6} "
      f"{'p1 gain':>7} {'p1-matched time%':>16}")
for o in out:
    print(f"{o['dataset']:24} {o['target']:5.2f} {o['k']:5d} {o['setting']} {o['method']:10} {o['mean_r']:.4f} "
          f"{fm(o['save_dc'])} {fm(o['save_lat'])} {fm(None if o['p1_gain'] is None else 100 * o['p1_gain'], 7)} "
          f"{fm(o['save_lat_p1'], 16)}")

def frac(cs, key, test):
    known = [c for c in cs if c[key] is not None]
    return f"{sum(test(c[key]) for c in known)}/{len(known)}"

print("\nCounts per operating point (same mean recall: less work / faster / p1 >= fixed;  same p1: faster)")
groups = defaultdict(list)
for o in out:
    side = "non-Gaussian" if (o["ks"] or 0) >= BAND_HI else "Gaussian+band"
    groups[(o["target"], o["k"], side, o["method"])].append(o)
summary = []
for (tgt, k, side, meth), cs in sorted(groups.items()):
    meet = sum(abs(c["mean_r"] - tgt) <= 0.01 for c in cs)
    worst = min((c["save_dc"] for c in cs if c["save_dc"] is not None), default=None)
    row = dict(target=tgt, k=k, side=side, method=meth, runs=len(cs),
               less_work=frac(cs, "save_dc", lambda v: v > 0), worst_dc=worst,
               faster=frac(cs, "save_lat", lambda v: v > 0), p1_ge_fixed=frac(cs, "p1_gain", lambda v: v >= 0),
               faster_p1=frac(cs, "save_lat_p1", lambda v: v > 0),
               median_time_p1=float(np.median([c["save_lat_p1"] for c in cs if c["save_lat_p1"] is not None] or [np.nan])),
               within_1pt_of_target=f"{meet}/{len(cs)}")
    summary.append(row)
    print(f"  target {tgt:.2f} k {k:5d} {side:14} {meth:10} runs {len(cs):2d}: less work {row['less_work']:>5} "
          f"(worst {fm(worst)}%), faster {row['faster']:>5}, p1>=fixed {row['p1_ge_fixed']:>5} | p1-matched faster "
          f"{row['faster_p1']:>5} (median {row['median_time_p1']:+.1f}%) | mean recall within 0.01 of target {row['within_1pt_of_target']}")

with open(os.path.join(root, "sweep_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
with open(os.path.join(root, "sweep_counts.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)
print(f"\nwrote {os.path.join(root, 'sweep_summary.csv')} and sweep_counts.csv")
