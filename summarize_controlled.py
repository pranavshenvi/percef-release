#!/usr/bin/env python3
"""
Collect every results_controlled_<config>_*/summary.json (latest run per
config) written by benchmark_controlled.py, print one table, and draw the two
figures the controlled experiments exist to produce:

  1. rho advantage vs. measured KS          (Experiment B + C): does a less
     Gaussian score distribution alone give our score a bigger edge?
  2. DC change at Ada-ef's recall vs. spread (Experiment A + C): does wider
     difficulty spread alone turn that edge into bigger savings?

The 8 real datasets (summary.md §7, Cohere excluded as its outcome is set by
the separate calibration-collapse mechanism) are drawn in grey for context.
Hollow markers are upper bounds: our cheapest run already beat Ada-ef, so the
true equal-recall saving is at least that large.

Usage:
  python summarize_controlled.py
"""

import glob, json, os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name: (KS, rho advantage, spread P90/Mean, DC change at Ada-ef recall %, is_upper_bound)
REAL_DATASETS = {
    "SIFT-128":      (0.125,  0.45,  1.48,   0.3, False),
    "Yambda audio":  (0.0856, 0.42,  1.08,  -4.4, True),
    "DeepImage-96":  (0.067,  0.26,  1.18,   0.7, True),
    "MS MARCO-384":  (0.047,  0.17,  3.22, -12.1, False),
    "dbpedia-1536":  (0.0388, 0.056, 1.82, -11.1, True),
    "LAION-I2I":     (0.029,  0.10,  1.51,   1.1, True),
    "GloVe-100":     (0.016,  0.02,  1.86,  -6.9, True),
}
EXP_COLOR = {"A": "#2a6f6f", "B": "#a3762a", "C": "#b5502e", "D": "#5b4a9e"}

def synth_label(s):
    if s["experiment"] == "D":
        return f"sep={s['synth_center_scale']}"
    return f"α={s['alpha']}"

def latest_summaries():
    by_config = {}
    for path in sorted(glob.glob("results_controlled_*/summary.json")):
        with open(path) as f:
            s = json.load(f)
        by_config[s["config"]] = s           # sorted by timestamp, so the last one wins
    return [by_config[k] for k in sorted(by_config)]

def main():
    runs = latest_summaries()
    if not runs:
        print("No results_controlled_*/summary.json found -- run benchmark_controlled.py first.")
        return

    hdr = (f"{'Config':<24} {'Exp':>3} {'KS':>7} {'rho Ada':>8} {'rho ours':>9} {'adv':>7} "
           f"{'Spread':>7} {'minEF P90/med':>13} {'@floor':>6} {'Ada DC':>8} {'Ada R':>7} {'dDC@R':>9} {'dDC@tgt':>9}")
    print(hdr)
    print("─" * len(hdr))
    for s in runs:
        def d(key):
            r = s.get(key)
            if r is None: return "never"
            return ("<=" if r["bound"] else "") + f"{r['delta_pct']:+.1f}%"
        print(f"{s['config']:<24} {s['experiment']:>3} {s['ks_mean']:>7.4f} {s['rho_ada']:>+8.3f} "
              f"{s['rho_ours_best']:>+9.3f} {s['rho_advantage']:>+7.3f} {s['spread_proxy_k1']:>6.2f}x "
              f"{s['calib_min_ef']['p90_over_median']:>12.2f}x {s['calib_min_ef'].get('frac_at_ef_floor', float('nan'))*100:>5.0f}% {s['ada']['dc']:>8.0f} {s['ada']['recall']:>7.4f} "
              f"{d('equal_recall'):>9} {d('equal_target'):>9}")

    out_dir = f"results_controlled_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    for name, (ks, adv, _, _, _) in REAL_DATASETS.items():
        ax.scatter(ks, adv, color="#b8b3a6", s=36, zorder=1)
        ax.annotate(name, (ks, adv), fontsize=7, color="#8a8578", xytext=(4, 3), textcoords="offset points")
    for s in runs:
        if s["experiment"] in ("B", "C", "D"):
            ax.scatter(s["ks_mean"], s["rho_advantage"], color=EXP_COLOR[s["experiment"]], s=60, zorder=3)
            ax.annotate(synth_label(s) + (" +hard" if s["hard_frac"] else ""), (s["ks_mean"], s["rho_advantage"]),
                        fontsize=8, xytext=(5, -10), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("Mean KS statistic (higher = less Gaussian)")
    ax.set_ylabel("ρ advantage  (|ρ ours| − |ρ Ada-ef|)")
    ax.set_title("Factor A: Gaussian fit → score quality")

    ax = axes[1]
    for name, (_, _, spread, ddc, bound) in REAL_DATASETS.items():
        ax.scatter(spread, ddc, s=36, zorder=1, facecolors="none" if bound else "#b8b3a6", edgecolors="#b8b3a6")
        ax.annotate(name, (spread, ddc), fontsize=7, color="#8a8578", xytext=(4, 3), textcoords="offset points")
    for s in runs:
        r = s.get("equal_recall")
        if s["experiment"] in ("A", "C", "D") and r is not None:
            c = EXP_COLOR[s["experiment"]]
            ax.scatter(s["spread_proxy_k1"], r["delta_pct"], s=60, zorder=3,
                       facecolors="none" if r["bound"] else c, edgecolors=c, linewidths=1.5)
            label = (synth_label(s) if s["experiment"] == "D" else f"{int(s['hard_frac']*100)}% hard" + (f" σ={s['hard_sigma']}" if s["hard_frac"] else "") + (f", α={s['alpha']}" if s["base"] == "synth" else ""))
            ax.annotate(label, (s["spread_proxy_k1"], r["delta_pct"]), fontsize=8, xytext=(5, -10), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("Difficulty spread (P90 / Mean calibrated ef, K=1)")
    ax.set_ylabel("DC change vs Ada-ef at equal recall (%)")
    ax.set_title("Factor C: spread → savings  (hollow = upper bound)")

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"Experiment {e}") for e, c in EXP_COLOR.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="#b8b3a6", label="Real datasets"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig_path = os.path.join(out_dir, "controlled_factors.png")
    fig.savefig(fig_path, dpi=150)

    with open(os.path.join(out_dir, "controlled_summary.json"), "w") as f:
        json.dump(runs, f, indent=2)
    print(f"\nWrote {fig_path} and {out_dir}/controlled_summary.json")

if __name__ == "__main__":
    main()
