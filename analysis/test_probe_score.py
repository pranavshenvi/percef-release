"""Check the running probe score (2026-10-06 latency fix).

search_percef now builds the PercEF score while the probe is collected instead of
copying the result heap afterwards. The calibration path (get_dynamic_probe_score_weighted)
still scores the distances in the heap. Both must pick the same ef for every query, and
the search must return the same neighbours as before; this script checks the first and
times the search against a fixed-ef search of similar cost.

    python analysis/test_probe_score.py
"""
import time
import numpy as np
import chao_hybrid_ada_ef_cpp as hnsw

rng = np.random.default_rng(0)
N, D, K, PROBE = 50_000, 96, 10, 100
# skewed, clustered data so the score spreads over several bins
centers = rng.normal(size=(50, D)).astype(np.float32) * 4
data = (centers[rng.integers(0, 50, N)] + rng.exponential(1.0, (N, D))).astype(np.float32)
queries = (centers[rng.integers(0, 50, 2000)] + rng.exponential(1.0, (2000, D))).astype(np.float32)

idx = hnsw.Index(space="l2", dim=D)
idx.init_index(max_elements=N, ef_construction=200, M=16)
idx.add_items(data)

# bins from the empirical probe distances of a few queries (as in the benchmark)
probe_d = np.sort(((data[rng.choice(N, 2000, replace=False)][None] - queries[:200, None]) ** 2).sum(-1).ravel())
bins = [float(np.quantile(probe_d, q)) for q in (0.001, 0.002, 0.003, 0.004, 0.005)]
weights = [float(100.0 * np.exp(-i)) for i in range(len(bins))]
ef_table = [200, 160, 120, 90, 60, 40, 30, 20, 15, 12, 10] + [10] * 90
MIN_EF, MAX_EF = K, 400
cfg = hnsw.PercEFConfig(bins, weights, ef_table, MIN_EF, MAX_EF, PROBE)

mismatch, efs = 0, []
for q in queries:
    _, _, ef_used = idx.search_percef(q, K, cfg)
    score = idx.get_dynamic_probe_score_weighted(q, bins, weights, PROBE)
    i = max(0, int(round(score)))
    ef_cal = ef_table[i] if i < len(ef_table) else ef_table[-1]
    ef_cal = max(MIN_EF, min(MAX_EF, ef_cal))
    mismatch += ef_used != ef_cal
    efs.append(ef_used)
print(f"ef from search vs calibration: {len(queries) - mismatch}/{len(queries)} identical "
      f"(ef used: min {min(efs)}, median {int(np.median(efs))}, max {max(efs)})")

def bench(fn, rounds=3):
    best = []
    for _ in range(rounds):
        t = time.perf_counter()
        for q in queries:
            fn(q)
        best.append((time.perf_counter() - t) / len(queries) * 1e6)
    return min(best)

ef_fixed = int(np.mean(efs))
idx.set_ef(ef_fixed)
t_fixed = bench(lambda q: idx.knn_query(q, k=K))
t_ours = bench(lambda q: idx.search_percef(q, K, cfg))
print(f"per query: PercEF {t_ours:.1f} us, fixed ef={ef_fixed} {t_fixed:.1f} us (sanity only; the benchmark is the real test)")
raise SystemExit(1 if mismatch else 0)
