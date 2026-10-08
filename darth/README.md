# DARTH comparison

DARTH (Chatzakis, Papakonstantinou, Palpanas, SIGMOD 2026) stops an HNSW search early when a
LightGBM model, fed with features of the search so far, predicts that the target recall is
reached. It is the learned baseline the Ada-ef paper compares against, so the PercEF paper needs it.

We run the authors' own code (their FAISS fork, pinned to commit `0d9bafc`) on our datasets, with
the same test queries and ground truth as PercEF and Ada-ef. Two non-algorithmic changes, both in
`setup_darth.sh`: the LightGBM path in CMake (`~` is not expanded) and a generic dataset layout
in the data loader. Search, features and predictor are untouched.

## Protocol
| | DARTH | PercEF / Ada-ef |
|---|---|---|
| Index | FAISS HNSW, M=16, efConstruction=500 | HNSWlib, M=16, efConstruction=500 |
| Training / calibration | R-calibration queries: 3/4 train the predictor, 1/4 (<=500) tune its prediction intervals | the same R-calibration queries |
| Test | the same test queries, same ground truth | same |
| Target | recall 0.95 | 0.95 |
| Predictor | LightGBM, n_estimators=100, their `all_feats` (as their `predictor_training.py`) | — |
| Intervals | their grid (ipi in D x {1,1/2,1/4,1/8}, mpi in D x {1,...,1/20}), chosen on validation | — |
| Threads | one (OMP_NUM_THREADS=1) | one |
| Reported against | FAISS fixed ef on the same index | HNSWlib fixed ef |

Each method is compared with a fixed ef **of its own library**: the two HNSW implementations build
different graphs and time queries differently (FAISS in C++, our harness through Python), so
absolute latencies across libraries are not compared. Recall, p1/p5 and target-hit share are
directly comparable. DARTH's paper trained on 10,000 queries; here it gets the same budget as the
other methods (about 1,500), which we state.

## Running (on the server, from the repo root, with nothing else running)
    sudo apt-get install -y cmake g++ libopenblas-dev libomp-dev     # once
    pip install --user lightgbm pandas                                # once
    bash darth/setup_darth.sh                                         # once, ~15-30 min
    bash darth/run_darth.sh glove100                                  # per dataset
Output: `results_darth_<dataset>_<ts>/` with `summary.json`, the per-query logs, the model and
`run.log`; the summary prints DARTH next to our methods on the same queries.
Memory: DARTH loads the whole corpus and keeps a second copy in the index (Cohere and LAION do not
fit in 62 GB; the 1-10M-vector datasets do).
