"""Why time and distance counts part ways for PercEF (updateAsOf071026.md).

Under a plain fixed ef, the queries PercEF calls hard (top 20% of the ef it assigns) take more time
per distance computation than the queries it calls easy (bottom 20%). PercEF moves work from easy to
hard queries, so the time it saves is smaller than the work it saves. This measures, per run:

  hard / easy : median of (query time / time a fixed-ef query with the same distance count takes),
                for those queries, under the fixed ef whose mean recall is closest to PercEF's
  ours        : the same ratio over all of PercEF's own queries
  hard DC     : share of PercEF's distance computations spent on its hard queries

Fixed-ef time at a given distance count: median time of all fixed-ef queries in 40 bins of log DC.
Needs per_query_<S>.npz (written by benchmark_unified.py).

    python analysis/query_cost.py server_results [dataset,dataset,...]
"""
import glob, os, re, sys
import numpy as np

OURS = "Ours (K=1, Isotonic)"


def run_cost(npz_path):
    z = np.load(npz_path)
    names = sorted({k.split("|")[0] for k in z.keys()})
    fx = [n for n in names if n.startswith("Fixed")]
    if OURS not in names or not fx or f"{OURS}|lat_us" not in z:
        return None
    dc = np.concatenate([z[n + "|dc"] for n in fx]).astype(float)
    lat = np.concatenate([z[n + "|lat_us"] for n in fx]).astype(float)
    ok = (dc > 0) & (lat > 0)
    dc, lat = dc[ok], lat[ok]
    edges = np.quantile(np.log(dc), np.linspace(0, 1, 41))
    b = np.clip(np.searchsorted(edges, np.log(dc)) - 1, 0, 39)
    keep = [i for i in range(40) if (b == i).any()]
    centers = np.array([np.median(np.log(dc)[b == i]) for i in keep])
    med = np.array([np.median(lat[b == i]) for i in keep])
    pred = lambda x: np.interp(np.log(np.maximum(x, 1)), centers, med)

    ef = z[OURS + "|ef"]
    hard, easy = ef >= np.quantile(ef, 0.8), ef <= np.quantile(ef, 0.2)
    ref = min(fx, key=lambda n: abs(np.mean(z[n + "|recall"]) - np.mean(z[OURS + "|recall"])))
    r_ref = z[ref + "|lat_us"] / pred(z[ref + "|dc"].astype(float))
    o_dc = z[OURS + "|dc"].astype(float)
    r_ours = z[OURS + "|lat_us"] / pred(o_dc)
    return dict(ref=ref, hard=float(np.median(r_ref[hard])), easy=float(np.median(r_ref[easy])),
                ours=float(o_dc @ r_ours / o_dc.sum()), hard_dc_share=float(o_dc[hard].sum() / o_dc.sum()))


def latest_runs(root):
    latest = {}
    for d in sorted(glob.glob(os.path.join(root, "results_unified_*"))):
        b = os.path.basename(d)
        if "smoke" in b or "ablation" in b or "_sweep" in b:
            continue
        m = re.match(r"results_unified_(.+)_(\d{8}_\d{6})$", b)
        if m and os.path.exists(os.path.join(d, "rows_R.json")):
            latest[m.group(1)] = d
    return latest


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = latest_runs(root)
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(runs)
    print(f"{'dataset':26} S  {'hard':>5} {'easy':>5} {'ours':>5} {'hard DC':>7}  reference")
    for ds in only:
        for S in "PR":
            f = os.path.join(runs[ds], f"per_query_{S}.npz")
            c = run_cost(f) if os.path.exists(f) else None
            if c:
                print(f"{ds:26} {S}  {c['hard']:5.3f} {c['easy']:5.3f} {c['ours']:5.3f} {100*c['hard_dc_share']:6.1f}%  {c['ref']}")
