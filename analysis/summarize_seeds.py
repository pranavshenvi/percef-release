"""Seed repeats (run_seeds.sh): how much do the paper's per-run numbers move with another random
draw of the R/test split, the P calibration points and the KS sample? The paper's own run is seed 42.

For each dataset, setting and method (PercEF default, Ada-ef as shipped): work saved, time saved and
p1 gain against the tuned fixed ef at the same mean recall (as make_paper_figures.py), and the rank
correlation of each score with the true ef. Prints every seed, then mean, standard deviation and
range, and whether the sign of each saving is the same for every seed. Writes seed_summary.csv.

    python analysis/summarize_seeds.py server_results
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_paper_figures import fixed_at   # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "."
METHODS = {"Ours (K=1, Isotonic)": "PercEF", "Ada-ef (as shipped)": "Ada-ef"}

seed_runs = defaultdict(dict)                  # ds -> seed -> dir
for d in sorted(glob.glob(os.path.join(root, "results_unified_*_sweep_s*_2*"))):
    m = re.match(r"results_unified_(.+)_sweep_s(\d+)_\d{8}_\d{6}$", os.path.basename(d))
    if m and os.path.exists(os.path.join(d, "rows_R.json")):
        seed_runs[m.group(1)][int(m.group(2))] = d
for ds in list(seed_runs):                     # the paper's run is seed 42
    base = [d for d in sorted(glob.glob(os.path.join(root, f"results_unified_{ds}_2*")))
            if re.match(rf"results_unified_{re.escape(ds)}_\d{{8}}_\d{{6}}$", os.path.basename(d))
            and os.path.exists(os.path.join(d, "rows_R.json"))]
    if base:
        seed_runs[ds][42] = base[-1]
if not seed_runs:
    sys.exit("no seed runs found (results_unified_<ds>_sweep_s<seed>_*); run run_seeds.sh first")

out = []
for ds, seeds in sorted(seed_runs.items()):
    for seed, d in sorted(seeds.items()):
        for S in ("P", "R"):
            f = os.path.join(d, f"rows_{S}.json")
            if not os.path.exists(f):
                continue
            rows = json.load(open(f))
            summ = json.load(open(os.path.join(d, f"summary_{S}.json"))) if os.path.exists(os.path.join(d, f"summary_{S}.json")) else {}
            fixed = sorted([r for r in rows if r["name"].startswith("Fixed")], key=lambda r: r["avg_ef"])
            for full, meth in METHODS.items():
                r = next((x for x in rows if x["name"] == full), None)
                if r is None:
                    continue
                fa = lambda key: fixed_at(fixed, r["mean_r"], key)
                sv = lambda key: None if fa(key) is None or r.get(key) is None else (fa(key) - r[key]) / fa(key) * 100
                rho = summ.get("rho_ada") if meth == "Ada-ef" else (summ.get("rho_ours") or {}).get("1")
                out.append(dict(dataset=ds, seed=seed, setting=S, method=meth, mean_r=r["mean_r"],
                                save_dc=sv("total_dc"), save_lat=sv("mean_lat_us"),
                                p1_gain=None if fa("p1") is None else 100 * (r["p1"] - fa("p1")),
                                rho=None if rho is None else abs(rho)))

fm = lambda v, w=6, d=1: " " * (w - 3) + "n/a" if v is None else f"{v:+{w}.{d}f}"
print(f"{'dataset':20} S {'method':7} {'seed':>4} {'recall':>7} {'work%':>6} {'time%':>6} {'p1 pts':>6} {'|rho|':>5}")
for o in out:
    print(f"{o['dataset']:20} {o['setting']} {o['method']:7} {o['seed']:>4} {o['mean_r']:7.4f} {fm(o['save_dc'])} "
          f"{fm(o['save_lat'])} {fm(o['p1_gain'])} {'  n/a' if o['rho'] is None else f'{o['rho']:5.2f}'}")

print("\nAcross seeds: mean +- sd [min, max]; 'stable' = same sign for every seed")
g = defaultdict(list)
for o in out:
    g[(o["dataset"], o["setting"], o["method"])].append(o)
summary = []
for (ds, S, meth), cs in sorted(g.items()):
    line = f"{ds:20} {S} {meth:7} n={len(cs)}"
    row = dict(dataset=ds, setting=S, method=meth, seeds=len(cs))
    for key in ("save_dc", "save_lat", "p1_gain", "rho"):
        v = np.array([c[key] for c in cs if c[key] is not None], float)
        if not len(v):
            continue
        stable = bool(np.all(v > 0) or np.all(v < 0)) if key != "rho" else None
        row.update({f"{key}_mean": v.mean(), f"{key}_sd": v.std(ddof=1) if len(v) > 1 else 0.0,
                    f"{key}_min": v.min(), f"{key}_max": v.max(), f"{key}_stable": stable})
        line += (f" | {key} {v.mean():+.2f}+-{row[key + '_sd']:.2f} [{v.min():+.2f},{v.max():+.2f}]"
                 + ("" if stable is None else (" stable" if stable else " SIGN CHANGES")))
    summary.append(row)
    print(line)

with open(os.path.join(root, "seed_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in summary for k in r}, key=lambda k: (k not in ("dataset", "setting", "method", "seeds"), k)))
    w.writeheader(); w.writerows(summary)
print(f"\nwrote {os.path.join(root, 'seed_summary.csv')}")
