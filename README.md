# PercEF: adaptive HNSW search beyond the Gaussian assumption

Code, scripts and result files for the paper *"PercEF: Exploiting Empirical Percentiles for
Adaptive HNSW Search Beyond the Gaussian Assumption"*. Every table and figure in the paper is
produced from the files in `results/` by the scripts below.

## What is here

| Path | Contents |
|---|---|
| `chao_hybrid_ada_ef/` | HNSWlib with Ada-ef (Zhang & Miller, SIGMOD 2026; Apache-2.0, original README in `ADA_EF_README.md`) and our additions: PercEF's search (`hnswlib/hnswalg.h`, `searchKnnDynamicWeighted`), its configuration object and Python bindings (`python_bindings/bindings.cpp`). Ada-ef's scoring, sketch and search are its authors' code, unchanged, plus a loader for streamed statistics. Eigen headers (MPL-2.0) are included because Ada-ef's code needs them. |
| `benchmark_unified.py` | The end-to-end benchmark: fixed-ef grid, Ada-ef (as shipped and with its WAE floor), PercEF; settings P and R; work, latency (3 rotated rounds), recall, p1/p5. Flags for the probe-length ablation, the robustness sweep (`--target-recall`, `--k`) and the two fixes (`--fixes`). |
| `survey_ks_vibe.py`, `survey_ks_standard.py`, `survey_tail.py` | The KS survey over 41 datasets and the tail-weighted alternatives |
| `benchmark_controlled.py`, `controlled_datasets.py`, `summarize_controlled.py`, `run_controlled_experiments.sh` | Controlled synthetic data |
| `darth/` | DARTH and LAET (SIGMOD 2026, SIGMOD 2020) through DARTH's authors' FAISS fork, on the same queries; see `darth/README.md` |
| `analysis/` | `make_paper_figures.py` (all figures and tables), `rescore_scorecard.js` (independent recount of the counts), `query_cost.py`, `summarize_sweep.py`, `test_probe_score.py` |
| `run_*.sh` | The run scripts used on the server (ablation + DARTH, re-timing, sweep, fixes, learned baselines) |
| `results/` | Result files of every run used in the paper (per-run `rows_*.json`, `summary_*.json`, `meta.json`, and per-query `per_query_*.npz` for the main runs) |
| `paper_outputs/` | The figures and tables exactly as produced for the paper |

## Setup

Linux, Python 3.10+, a C++17 compiler.

    pip install numpy scipy scikit-learn h5py matplotlib pybind11 requests tqdm
    cd chao_hybrid_ada_ef && python setup.py build_ext --inplace && cd ..
    python analysis/test_probe_score.py        # sanity check of the PercEF search (same ef as calibration)

Datasets: `python download_unified_data.py` and the `--download-only` options of the two survey
scripts fetch the public files (ann-benchmarks, Big-ANN prefixes, VIBE, Yambda, and the Ada-ef
paper's sources). Cohere and LAION are used as subsets that fit in 62 GB of memory.

## Reproducing the paper

From the included results (minutes, no datasets needed):

    python analysis/make_paper_figures.py --results-dir results     # all figures and tables -> paper_out_<ts>/
    node analysis/rescore_scorecard.js results --loo                # recount of every count in the paper
    python analysis/query_cost.py results                           # time per distance, hard vs easy queries
    python analysis/summarize_sweep.py results                      # robustness sweep and fixes
    python darth/darth_summarize.py results/results_darth_<dataset>_<ts>   # DARTH / LAET vs fixed ef

| Paper item | Produced by |
|---|---|
| KS survey figure, KS table | `make_paper_figures.py` (`fig1_ks_survey`, `table2_ks_survey`) from the survey results |
| Tail recall vs KS (teaser) | `fig2_p1_vs_ks` |
| Datasets table | `table1_datasets` |
| Counts table (work, time, p1, p1-matched) | `table3b_counts`; recounted by `rescore_scorecard.js` |
| Per-run scorecard | `table3_scorecard` |
| Cost and latency vs recall, latency CDF | `fig3_cost_vs_recall`, `fig3b_latency_vs_recall`, `fig3c_latency_cdf` |
| Per-query cost (hard vs easy queries) | `table3c_query_cost` (`query_cost.py`) |
| Probe-length ablation | `results_unified_*_ablation_*` (`benchmark_unified.py --ablation`) |
| Robustness sweep and fixes | `summarize_sweep.py` over `results_unified_*_sweep_*` |
| DARTH / LAET table | `darth_summarize.py` over `results_darth_*` |
| Offline cost | `table4_offline` |
| Tail-weighted alternatives | `table5_tail_survey` |
| Controlled data | `fig4_controlled` |

From scratch (hours per dataset, one machine, one search thread):

    python benchmark_unified.py --dataset sift128                       # both settings, P then R
    python benchmark_unified.py --dataset sift128 --ablation --quick --settings R
    python benchmark_unified.py --dataset sift128 --quick --target-recall 0.99 --fixes
    bash darth/setup_darth.sh && bash darth/run_darth.sh sift128 && bash darth/run_laet.sh sift128

`benchmark_unified.py --help` lists all datasets. Protocol (frozen): M = 16, efConstruction = 500,
target recall 0.95, ef cap 5000, k = 100 (1000 for MS MARCO, Cohere, LAION), PercEF with one
centroid, 5 bins at the 0.1-0.5% quantiles, a 100-distance probe and an isotonic ef table.

## License

Apache-2.0 for our code, as for the Ada-ef code it extends; Eigen under its own licenses
(`chao_hybrid_ada_ef/Eigen/COPYING.*`).
