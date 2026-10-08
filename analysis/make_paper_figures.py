#!/usr/bin/env python3
"""
Paper tables and figures, generated only from result files (PAPER_PLAN.md, "Deliverables").

Reads, from --results-dir (default: the repo root on the server):
  results_unified_<dataset>_<ts>/   rows_{P,R}.json, summary_{P,R}.json, meta.json, per_query_{P,R}.npz
                                    (latest non-smoke run per dataset)
  results_ks_survey_*/ks_survey.json        KS surveys (local, VIBE, standard; later runs override)
  results_tail_survey_*/tail_survey.json    tail survey (latest), table 5 only
  results_controlled_summary_*/controlled_summary.json   controlled experiments (latest)

Writes paper_out_<ts>/ with CSV + LaTeX (booktabs) tables and PDF + PNG figures:
  table1_datasets       datasets, sizes, KS (200 q), calibration rho of both scores
  table2_ks_survey      KS with 95% interval for every surveyed dataset
  table3_scorecard      each method vs a tuned fixed ef at the same mean recall: DC and latency
                        saving, p1/p5 gain, whether it adapts (one ef for all queries = no)
  table3b_counts        the counts quoted in the paper (cheaper, faster, p1 >= fixed, and against
                        the fixed ef that matches each method's p1)
  table3c_query_cost    time per distance of hard vs easy queries under a fixed ef (query_cost.py)
  table4_offline        offline time and memory (runs that recorded them)
  table5_tail_survey    tail-weighted alternatives to KS
  fig1_ks_survey        KS of every surveyed dataset, crossover band shaded
  fig2_p1_vs_ks         p1 gain over fixed ef vs KS, ours and Ada-ef
  fig3_cost_vs_recall   DC vs mean recall per dataset (setting R): fixed-ef curve, methods as points
  fig3b_latency_vs_recall  the same in wall-clock latency, runs that recorded it
  fig3c_latency_cdf     per-query latency CDF, two datasets
  fig4_controlled       controlled experiment: rho advantage vs KS, with the real datasets

"Tuned fixed ef at recall r" is interpolated between the two fixed-ef grid points around r in
log(1 - recall) vs log(cost) (updateAsOf290926.md §1), exactly as analysis/rescore_scorecard.js;
the counts printed at the end must match that script's.

Usage:  python3 analysis/make_paper_figures.py [--results-dir .]
"""

import os, re, csv, glob, json, argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical slots 1-3 (dataviz reference palette, light mode) + neutral reference
OURS_C, ADA_C, WAE_C, FIXED_C = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, BAND_C = "#0b0b0b", "#52514e", "#e6e5e1", "#efeeea"
METHODS = {"Ours (K=1, Isotonic)": "Ours", "Ada-ef (as shipped)": "Ada-ef", "Ada-ef (WAE floor)": "Ada-ef WAE"}
COLOR = {"Ours": OURS_C, "Ada-ef": ADA_C, "Ada-ef WAE": WAE_C}
MARKER = {"Ours": "o", "Ada-ef": "s", "Ada-ef WAE": "D"}     # shape as well as colour: never colour alone
SIZE = {"Ours": 30, "Ada-ef": 26, "Ada-ef WAE": 20}
CONNECT_C = "#c9c8c2"
BAND_LO, BAND_HI = 0.044, 0.0655   # edges measured on MS MARCO-384 (0.0442) and DeepImage-96 (0.0656), quoted
                                    # as 0.044-0.066; 0.0655 keeps DeepImage on its own (upper) edge after rounding
CROSS_MODAL = {"vibe_imagenet_align", "coco_t2i", "lastfm64"}
NAMES = {"glove100": "GloVe-100", "deepimage96": "DeepImage-96", "sift128": "SIFT-1M", "dbpedia1536": "DBpedia-1536",
         "yambda": "Yambda", "msmarco384": "MS MARCO-384", "cohere1024": "Cohere-1024", "laion_i2i": "LAION-I2I",
         "vibe_landmark_dino": "Landmark-DINO", "vibe_inaturalist_resnet": "iNat-ResNet",
         "vibe_yahoo_minilm": "Yahoo-MiniLM", "vibe_imagenet_align": "ImageNet-ALIGN", "gist960": "GIST-960",
         "fashionmnist784": "Fashion-MNIST", "lastfm64": "Last.fm", "deep1b": "Deep1B (2M)", "coco_i2i": "COCO-I2I",
         "coco_t2i": "COCO-T2I", "bigann": "BIGANN (2M)", "msturing": "MS Turing (2M)"}
SOURCE = {"glove100": "ann-benchmarks", "deepimage96": "ann-benchmarks", "sift128": "ann-benchmarks",
          "dbpedia1536": "DBpedia, OpenAI embeddings", "yambda": "Yambda", "msmarco384": "MS MARCO (MiniLM)",
          "cohere1024": "Cohere, files 00-04", "laion_i2i": "LAION, shards 0-19",
          "vibe_landmark_dino": "VIBE", "vibe_inaturalist_resnet": "VIBE", "vibe_yahoo_minilm": "VIBE",
          "vibe_imagenet_align": "VIBE", "gist960": "ann-benchmarks", "fashionmnist784": "ann-benchmarks",
          "lastfm64": "ann-benchmarks", "coco_i2i": "ann-benchmarks", "coco_t2i": "ann-benchmarks",
          "deep1b": "Big-ANN", "bigann": "Big-ANN", "msturing": "Big-ANN"}
IN_ADA_PAPER = {"glove100": "yes", "deepimage96": "yes", "cohere1024": "yes (subset)", "laion_i2i": "yes (subset)",
                "msmarco384": "stand-in"}
# KS of record (200-query surveys, as quoted in the docs; DeepImage 0.0656 is the band edge itself, so it is
# listed as 0.066). Grouping and tables use these; the tail survey re-measurement appears only in table 5.
KS = {"glove100": .018, "dbpedia1536": .038, "msmarco384": .044, "cohere1024": .049, "laion_i2i": .034,
               "deepimage96": .066, "yambda": .090, "sift128": .126, "vibe_landmark_dino": .076,
               "vibe_inaturalist_resnet": .115, "gist960": .091, "fashionmnist784": .073, "vibe_yahoo_minilm": .054,
               "vibe_imagenet_align": .058, "lastfm64": .216, "deep1b": .067, "coco_i2i": .046, "coco_t2i": .056,
               "bigann": .029, "msturing": .010}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(re.sub(r"\bNaN\b", "null", f.read()))


def latest_per_name(root, prefix):
    """prefix<name>_<YYYYmmdd_HHMMSS> -> latest folder per name (smoke runs skipped)."""
    out = {}
    for d in glob.glob(os.path.join(root, prefix + "*")):
        b = os.path.basename(d)
        m = re.match(re.escape(prefix) + r"(.+)_(\d{8}_\d{6})$", b)
        if not m or "smoke" in b or "ablation" in b or "_sweep" in b or not os.path.isdir(d):
            continue
        if m.group(1) not in out or b > os.path.basename(out[m.group(1)]):
            out[m.group(1)] = d
    return out


def latest_dir(root, pattern):
    ds = sorted(d for d in glob.glob(os.path.join(root, pattern)) if os.path.isdir(d))
    return ds[-1] if ds else None


def fixed_at(fixed, r, key):
    """Tuned fixed ef at mean recall r: log(1 - recall) vs log(cost) interpolation; p1/p5 linear in t."""
    x = lambda v: np.log(max(1.0 - v, 1e-6))
    for a, c in zip(fixed, fixed[1:]):
        if a["mean_r"] < r <= c["mean_r"]:
            if a.get(key) is None or c.get(key) is None:
                return None
            t = (x(r) - x(a["mean_r"])) / (x(c["mean_r"]) - x(a["mean_r"]))
            if key in ("p1", "p5"):
                return a[key] + t * (c[key] - a[key])
            return float(np.exp(np.log(a[key]) + t * np.log(c[key] / a[key])))
    return None


def fixed_at_p1(fixed, p1, key):
    """Cheapest fixed ef reaching worst-1% recall p1: walk the grid in order of cost, keep the best p1 so far,
    interpolate log cost linearly in p1. None if no grid point reaches p1 (updateAsOf071026.md)."""
    fx = sorted(fixed, key=lambda q: q["total_dc"])
    if not fx or any(q.get(key) is None for q in fx):
        return None
    best = np.maximum.accumulate([q["p1"] for q in fx])
    if p1 <= best[0]:
        return fx[0][key]
    for i in range(1, len(fx)):
        if best[i - 1] < p1 <= best[i]:
            t = (p1 - best[i - 1]) / (best[i] - best[i - 1])
            return float(np.exp(np.log(fx[i - 1][key]) + t * np.log(fx[i][key] / fx[i - 1][key])))
    return None


def write_table(out, name, headers, rows, fmt=None):
    """CSV with raw values, LaTeX (booktabs) with formatted ones."""
    with open(os.path.join(out, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    fmt = fmt or {}
    esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    def cell(h, v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "--"
        return esc(fmt[h](v) if h in fmt else v)
    numeric = lambda j: all(r[j] is None or isinstance(r[j], (int, float)) for r in rows)
    align = "".join("r" if rows and numeric(j) else "l" for j in range(len(headers)))
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(esc(h) for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(h, v) for h, v in zip(headers, r)) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def shade_band(ax, vertical=True):
    (ax.axvspan if vertical else ax.axhspan)(BAND_LO, BAND_HI, color=BAND_C, zorder=0, lw=0)


pct = lambda v: f"{v:+.1f}%"
gain = lambda v: f"{v:+.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=".")
    args = ap.parse_args()
    R = args.results_dir
    out = f"paper_out_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out, exist_ok=True)

    runs = latest_per_name(R, "results_unified_")
    tail_dir = latest_dir(R, "results_tail_survey_*")
    tail = read_json(os.path.join(tail_dir, "tail_survey.json")) if tail_dir else {}
    ks_of = {ds: KS.get(ds) for ds in runs}
    print(f"{len(runs)} benchmark datasets; tail survey: {tail_dir or 'not found (table 5 skipped)'}")

    # ---- load every run -------------------------------------------------------------------
    data = {}
    for ds, d in runs.items():
        meta = read_json(os.path.join(d, "meta.json")) if os.path.exists(os.path.join(d, "meta.json")) else {}
        per = {}
        for S in ("P", "R"):
            rf = os.path.join(d, f"rows_{S}.json")
            if not os.path.exists(rf):
                continue
            rows = read_json(rf)
            sf = os.path.join(d, f"summary_{S}.json")
            per[S] = dict(rows=rows, summary=read_json(sf) if os.path.exists(sf) else {},
                          fixed=sorted([r for r in rows if r["name"].startswith("Fixed")], key=lambda r: r["avg_ef"]))
        data[ds] = dict(dir=d, meta=meta, settings=per)
    order = sorted(data, key=lambda ds: (ks_of[ds] if ks_of[ds] is not None else 9, ds))

    # ---- scorecard ------------------------------------------------------------------------
    card = []
    for ds in order:
        for S, s in data[ds]["settings"].items():
            for full, m in METHODS.items():
                r = next((x for x in s["rows"] if x["name"] == full), None)
                if r is None:
                    continue
                f = lambda key: fixed_at(s["fixed"], r["mean_r"], key)
                sv = lambda key: None if f(key) is None or r.get(key) is None else (f(key) - r[key]) / f(key) * 100
                gn = lambda key: None if f(key) is None else r[key] - f(key)
                g1 = lambda key: fixed_at_p1(s["fixed"], r["p1"], key)
                sv1 = lambda key: None if g1(key) is None or r.get(key) is None else (g1(key) - r[key]) / g1(key) * 100
                card.append(dict(dataset=ds, setting=S, method=m, ks=ks_of[ds], cross_modal=ds in CROSS_MODAL,
                                 mean_r=r["mean_r"], adapts=None if r.get("distinct_ef") is None else r["distinct_ef"] > 1,
                                 save_dc=sv("total_dc"), save_lat=sv("mean_lat_us"), p1_gain=gn("p1"), p5_gain=gn("p5"),
                                 save_dc_p1=sv1("total_dc"), save_lat_p1=sv1("mean_lat_us"),
                                 lat_us=r.get("mean_lat_us"), dc=r["total_dc"]))
    write_table(out, "table3_scorecard",
                ["dataset", "setting", "method", "KS", "mean recall", "DC saving", "latency saving", "p1 gain",
                 "p5 gain", "adapts"],
                [[NAMES.get(c["dataset"], c["dataset"]), c["setting"], c["method"], c["ks"], c["mean_r"], c["save_dc"],
                  c["save_lat"], c["p1_gain"], c["p5_gain"], {True: "yes", False: "no", None: ""}[c["adapts"]]]
                 for c in card],
                {"KS": lambda v: f"{v:.3f}", "mean recall": lambda v: f"{v:.4f}", "DC saving": pct,
                 "latency saving": pct, "p1 gain": gain, "p5 gain": gain})

    # ---- the counts quoted in the paper ----------------------------------------------------
    groups = [("all, same modality", lambda c: not c["cross_modal"]),
              (f"KS >= {BAND_HI}", lambda c: not c["cross_modal"] and c["ks"] >= BAND_HI),
              (f"band {BAND_LO}-{BAND_HI}", lambda c: not c["cross_modal"] and BAND_LO <= c["ks"] < BAND_HI),
              (f"KS < {BAND_LO}", lambda c: not c["cross_modal"] and c["ks"] < BAND_LO),
              ("cross-modal", lambda c: c["cross_modal"])]
    def frac(cs, key, test):
        known = [c for c in cs if c[key] is not None]
        return f"{sum(test(c[key]) for c in known)}/{len(known)}"
    count_rows = []
    print("\nCounts (log-log interpolation; must match analysis/rescore_scorecard.js)")
    for g, sel in groups:
        for m in ("Ours", "Ada-ef", "Ada-ef WAE"):
            cs = [c for c in card if c["method"] == m and sel(c)]
            if not cs:
                continue
            worst = min((c["save_dc"] for c in cs if c["save_dc"] is not None), default=None)
            row = [g, m, len(cs), frac(cs, "save_dc", lambda v: v > 0), worst, frac(cs, "save_lat", lambda v: v > 0),
                   frac(cs, "p1_gain", lambda v: v >= 0),
                   frac(cs, "save_dc_p1", lambda v: v > 0), frac(cs, "save_lat_p1", lambda v: v > 0),
                   float(np.median([c["save_lat_p1"] for c in cs if c["save_lat_p1"] is not None] or [np.nan]))]
            count_rows.append(row)
            print(f"  {g:<22} {m:<11} runs {len(cs):>2}  cheaper {row[3]:>6} (worst {'n/a' if worst is None else pct(worst)})"
                  f"  faster {row[5]:>5}  p1>=fixed {row[6]:>6}  | equal p1: cheaper {row[7]:>6} faster {row[8]:>6}"
                  f" (median {pct(row[9])})")
    write_table(out, "table3b_counts", ["group", "method", "runs", "cheaper than fixed (DC)", "worst DC saving",
                                        "faster than fixed (latency)", "p1 >= fixed", "cheaper at equal p1",
                                        "faster at equal p1", "median time saving at equal p1"], count_rows,
                {"worst DC saving": pct, "median time saving at equal p1": pct})

    # ---- per-query cost: hard queries cost more time per distance (analysis/query_cost.py) --------
    import query_cost
    qc_rows = []
    for ds in order:
        for S in ("P", "R"):
            f = os.path.join(data[ds]["dir"], f"per_query_{S}.npz")
            c = query_cost.run_cost(f) if os.path.exists(f) else None
            if c:
                qc_rows.append([NAMES.get(ds, ds), S, ks_of[ds], c["hard"], c["easy"], c["ours"], 100 * c["hard_dc_share"]])
    write_table(out, "table3c_query_cost", ["dataset", "setting", "KS", "hard / fixed-ef time at same DC",
                                            "easy / fixed-ef time at same DC", "PercEF / fixed-ef time at same DC",
                                            "PercEF DC on hard queries (%)"], qc_rows,
                {"KS": lambda v: f"{v:.3f}", "hard / fixed-ef time at same DC": lambda v: f"{v:.3f}",
                 "easy / fixed-ef time at same DC": lambda v: f"{v:.3f}",
                 "PercEF / fixed-ef time at same DC": lambda v: f"{v:.3f}",
                 "PercEF DC on hard queries (%)": lambda v: f"{v:.1f}"})

    # ---- table 1: datasets ----------------------------------------------------------------
    t1 = []
    for ds in order:
        mt, st = data[ds]["meta"], data[ds]["settings"]
        rho = lambda S, k: (st.get(S, {}).get("summary", {}).get("rho_ada") if k == "ada"
                            else (st.get(S, {}).get("summary", {}).get("rho_ours") or {}).get("1"))
        t1.append([NAMES.get(ds, ds), SOURCE.get(ds, ""), IN_ADA_PAPER.get(ds, "no"), mt.get("n_corpus"), mt.get("dim"),
                   mt.get("K"), mt.get("n_test"), ks_of[ds], rho("P", "ada"), rho("P", "ours"), rho("R", "ada"),
                   rho("R", "ours")])
    r3 = lambda v: f"{v:+.2f}"
    write_table(out, "table1_datasets", ["dataset", "source", "in Ada-ef paper", "corpus", "dim", "K", "test queries",
                                         "KS", "rho Ada-ef P", "rho ours P", "rho Ada-ef R", "rho ours R"], t1,
                {"KS": lambda v: f"{v:.3f}", "corpus": lambda v: f"{v:,}", "rho Ada-ef P": r3, "rho ours P": r3,
                 "rho Ada-ef R": r3, "rho ours R": r3})

    # ---- table 2 / figure 1: KS survey ----------------------------------------------------
    survey = {}
    for d in sorted(glob.glob(os.path.join(R, "results_ks_survey_*"))):
        f = os.path.join(d, "ks_survey.json")
        if os.path.exists(f):
            for name, r in read_json(f).items():
                if isinstance(r, dict) and "ks" in r:
                    survey[name] = r                    # later runs override earlier ones
    if survey:
        items = sorted(survey.items(), key=lambda kv: kv[1]["ks"])
        write_table(out, "table2_ks_survey", ["dataset", "suite / split", "KS", "95% interval", "predicted"],
                    [[n, r.get("suite") or r.get("split", ""), r["ks"], r.get("ks_ci95"), r.get("predicted", "")]
                     for n, r in items],
                    {"KS": lambda v: f"{v:.4f}", "95% interval": lambda v: f"±{v:.4f}"})
        fig, ax = plt.subplots(figsize=(4.2, 0.13 * len(items) + 0.8))
        shade_band(ax)
        y = np.arange(len(items))
        ax.errorbar([r["ks"] for _, r in items], y, xerr=[r.get("ks_ci95") or 0 for _, r in items], fmt="o", ms=3,
                    color=INK, ecolor=INK2, elinewidth=0.8, capsize=0)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in items], fontsize=5.5)
        ax.set_xscale("log")
        ax.set_xlabel("KS against Ada-ef's CLT Normal (200 queries, ±95%)")
        ax.grid(axis="y", visible=False)
        ax.text(np.sqrt(BAND_LO * BAND_HI), len(items) - 0.2, "crossover band", ha="center", va="bottom", fontsize=6,
                color=INK2)
        ax.set_title(f"KS survey: {len(items)} datasets, "
                     f"{sum(r['ks'] >= BAND_HI for _, r in items)} on the non-Gaussian side of the band", loc="left")
        save(fig, out, "fig1_ks_survey")
        print(f"\nKS survey: {len(items)} datasets ({sum(r['ks'] >= BAND_HI for _, r in items)} >= {BAND_HI})")
    else:
        print("\nKS survey: no results_ks_survey_* folders found; table 2 / figure 1 skipped")

    # ---- figure 2: p1 gain over fixed ef, one row per dataset sorted by KS -------------------
    rows2 = []
    for ds in order:
        g = {m: [c["p1_gain"] for c in card if c["dataset"] == ds and c["method"] == m and c["p1_gain"] is not None]
             for m in ("Ours", "Ada-ef")}
        if g["Ours"] and g["Ada-ef"]:
            rows2.append((ds, float(np.mean(g["Ours"])), float(np.mean(g["Ada-ef"]))))
    fig, ax = plt.subplots(figsize=(4.4, 0.19 * len(rows2) + 0.9))
    for i, (ds, _, _) in enumerate(rows2):
        if BAND_LO <= ks_of[ds] < BAND_HI:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND_C, lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    for i, (_, o, a) in enumerate(rows2):
        ax.plot([a, o], [i, i], color=CONNECT_C, lw=1.5, zorder=2, solid_capstyle="round")
    for m, idx in (("Ada-ef", 2), ("Ours", 1)):
        for cm in (False, True):
            pts = [(r[idx], i) for i, r in enumerate(rows2) if (r[0] in CROSS_MODAL) == cm]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=MARKER[m], s=SIZE[m], zorder=3,
                           linewidths=1.2, facecolors="none" if cm else COLOR[m], edgecolors=COLOR[m],
                           label=m + (" (cross-modal: no adaptation)" if cm else ""))
    ax.set_yticks(np.arange(len(rows2)))
    ax.set_yticklabels([f"{NAMES.get(ds, ds)}  {ks_of[ds]:.3f}" for ds, _, _ in rows2], fontsize=6)
    ax.set_ylim(-0.6, len(rows2) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("p1 recall gain over a tuned fixed ef at equal mean recall (mean of P, R)")
    ax.set_xlim(right=ax.get_xlim()[1] + 0.01)
    ax.set_ylabel("dataset and KS (least Gaussian at top; shaded: band)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6)
    save(fig, out, "fig2_p1_vs_ks")

    # ---- figure 3: cost vs recall, small multiples -----------------------------------------
    def small_multiples(key, name, ylabel, only_with):
        dss = [ds for ds in order if "R" in data[ds]["settings"]
               and all(r.get(key) is not None for r in data[ds]["settings"]["R"]["fixed"])
               and (not only_with or any(r.get(key) is not None for r in data[ds]["settings"]["R"]["rows"]
                                         if r["name"] in METHODS))]
        if not dss:
            print(f"  {name}: no runs with {key}; skipped")
            return
        nc = 4 if len(dss) > 6 else 3
        nr = int(np.ceil(len(dss) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(2.2 * nc, 1.8 * nr), squeeze=False)
        for ax, ds in zip(axes.flat, dss):
            s = data[ds]["settings"]["R"]
            fx = [r for r in s["fixed"] if r["mean_r"] >= 0.80]
            ax.plot([r["mean_r"] for r in fx], [r[key] for r in fx], color=FIXED_C, lw=1.5, marker="o", ms=2.5,
                    label="fixed ef", zorder=2)
            for full, m in reversed(list(METHODS.items())):          # ours drawn last, on top
                r = next((x for x in s["rows"] if x["name"] == full), None)
                if r is not None and r.get(key) is not None:
                    ax.scatter([r["mean_r"]], [r[key]], s=SIZE[m], marker=MARKER[m], color=COLOR[m],
                               edgecolors="white", linewidths=1.0, zorder=4, label=m)
            ax.set_yscale("log")
            ax.set_title(f"{NAMES.get(ds, ds)}{' (x-modal)' if ds in CROSS_MODAL else ''} · KS {ks_of[ds]:.3f}",
                         loc="left", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.flat[len(dss):]:
            ax.set_visible(False)
        hl = {}
        for ax in axes.flat[:len(dss)]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                hl.setdefault(l, h)
        labels = [l for l in ("fixed ef", "Ours", "Ada-ef", "Ada-ef WAE") if l in hl]
        fig.legend([hl[l] for l in labels], labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel("mean recall@K (setting R)", fontsize=8)
        fig.supylabel(ylabel, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, out, name)
    small_multiples("total_dc", "fig3_cost_vs_recall", "distance computations per query", False)
    small_multiples("mean_lat_us", "fig3b_latency_vs_recall", "latency per query (µs)", True)

    # ---- figure 3c: per-query latency CDF ---------------------------------------------------
    lat_ds = [ds for ds in ("deep1b", "msturing", "bigann", "coco_i2i") if ds in data
              and os.path.exists(os.path.join(data[ds]["dir"], "per_query_R.npz"))
              and any(r.get("mean_lat_us") for r in data[ds]["settings"].get("R", {}).get("rows", []))][:2]
    if lat_ds:
        fig, axes = plt.subplots(1, len(lat_ds), figsize=(3.2 * len(lat_ds), 2.4), squeeze=False)
        for ax, ds in zip(axes.flat, lat_ds):
            z = np.load(os.path.join(data[ds]["dir"], "per_query_R.npz"))
            s = data[ds]["settings"]["R"]
            ours = next(r for r in s["rows"] if r["name"] == "Ours (K=1, Isotonic)")
            ref = next((r for r in s["fixed"] if r["mean_r"] >= ours["mean_r"]), s["fixed"][-1])
            for label, key, color in ((f"fixed ef={int(ref['avg_ef'])}", ref["name"], FIXED_C),
                                      ("Ada-ef", "Ada-ef (as shipped)", ADA_C), ("Ours", "Ours (K=1, Isotonic)", OURS_C)):
                k = f"{key}|lat_us"
                if k in z.files:
                    v = np.sort(z[k])
                    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=1.5, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("latency per query (µs)")
            ax.set_ylabel("share of queries")
            ax.set_title(NAMES.get(ds, ds) + " (setting R)", loc="left")
            ax.legend(loc="upper left")
        fig.tight_layout()
        save(fig, out, "fig3c_latency_cdf")
    else:
        print("  fig3c_latency_cdf: no run with per-query latency; skipped")

    # ---- table 4: offline cost --------------------------------------------------------------
    t4 = []
    for ds in order:
        for S, s in data[ds]["settings"].items():
            o = s["summary"].get("offline")
            if not o:
                continue
            a, k1 = o["ada"], o["ours"].get("K=1", {})
            t4.append([NAMES.get(ds, ds), S, a.get("statistics_s"), a.get("table_s"), a.get("memory_bytes"),
                       k1.get("bins_s"), k1.get("min_ef_s"), k1.get("memory_bytes")])
    if t4:
        sec = lambda v: f"{v:.1f}"
        write_table(out, "table4_offline", ["dataset", "setting", "Ada-ef statistics (s)", "Ada-ef table (s)",
                                            "Ada-ef memory (B)", "ours bins (s)", "ours min-ef (s)", "ours memory (B)"],
                    t4, {"Ada-ef statistics (s)": sec, "Ada-ef table (s)": sec, "ours bins (s)": sec,
                         "ours min-ef (s)": sec, "Ada-ef memory (B)": lambda v: f"{v:,}",
                         "ours memory (B)": lambda v: f"{v:,}"})

    # ---- table 5: tail survey ---------------------------------------------------------------
    if tail:
        cols = ["ks", "ad", "tailz_0.01", "tailz_0.001", "tailz_0.0001", "exc3", "margin"]
        write_table(out, "table5_tail_survey", ["dataset", "source"] + ["KS", "AD/n", "tail z 1e-2", "tail z 1e-3",
                                                                       "tail z 1e-4", "tail mass 99.9%", "rho margin"],
                    [[NAMES.get(ds, ds), SOURCE.get(ds, "")] + [tail[ds].get(c) for c in cols]
                     for ds in sorted(tail, key=lambda k: tail[k]["ks"])],
                    {"KS": lambda v: f"{v:.4f}", "AD/n": lambda v: f"{v:.4f}", "tail z 1e-2": lambda v: f"{v:+.2f}",
                     "tail z 1e-3": lambda v: f"{v:+.2f}", "tail z 1e-4": lambda v: f"{v:+.2f}",
                     "tail mass 99.9%": lambda v: f"{v:+.2f}", "rho margin": lambda v: f"{v:+.3f}"})

    # ---- figure 4: controlled experiment + real datasets ------------------------------------
    cdir = latest_dir(R, "results_controlled_summary_*")
    if cdir and os.path.exists(os.path.join(cdir, "controlled_summary.json")):
        runs_c = read_json(os.path.join(cdir, "controlled_summary.json"))
        runs_c = runs_c if isinstance(runs_c, list) else list(runs_c.values())
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        shade_band(ax)
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        real = [(ks_of[ds], np.mean([abs(s["summary"]["rho_ours"]["1"]) - abs(s["summary"]["rho_ada"])
                                     for s in data[ds]["settings"].values()
                                     if s["summary"].get("rho_ada") is not None
                                     and (s["summary"].get("rho_ours") or {}).get("1") is not None] or [np.nan]))
                for ds in order if ds not in CROSS_MODAL]
        real = [p for p in real if np.isfinite(p[1])]
        ax.scatter([p[0] for p in real], [p[1] for p in real], s=22, color=FIXED_C, zorder=2, label="real datasets")
        # synthetic runs by shape in ink: blue/orange mean "ours"/"Ada-ef" in every other figure
        for exp, marker, face, label in (("B", "^", INK, "synthetic B (α sweep)"),
                                         ("D", "s", "none", "synthetic D (cluster separation)")):
            pts = [(r["ks_mean"], r["rho_advantage"]) for r in runs_c
                   if r.get("experiment") == exp and r.get("ks_mean") is not None and r.get("rho_advantage") is not None]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=30, marker=marker, facecolors=face,
                           edgecolors=INK, linewidths=1.2, zorder=3, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("KS (higher = less Gaussian)")
        ax.set_ylabel("ranking advantage of our score\n|ρ ours| − |ρ Ada-ef|")
        ax.legend(loc="upper left")
        save(fig, out, "fig4_controlled")
    else:
        print("  fig4_controlled: no results_controlled_summary_* folder; skipped")

    print(f"\nWrote {out}/ ({len(os.listdir(out))} files)")


if __name__ == "__main__":
    main()
