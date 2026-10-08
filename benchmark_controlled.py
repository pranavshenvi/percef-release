#!/usr/bin/env python3
"""
Benchmark: controlled-factor experiments (see controlled_datasets.py).

Runs one config end to end with the same pipeline as benchmark_sift128.py
(K=100 group, target recall 0.95, M=16 / ef_construction=500, fixed fused
search path from the 2026-09-17 hnswalg.h fix), and additionally MEASURES the
three predictive factors for that config the same way they were measured on
the 8 real datasets, so the controlled runs sit on the same axes:

  - KS fit (Factor A):   diagnose_anisotropy.normality_check on the config's
                         own test queries (30 queries x 20,000 corpus points,
                         mean KS statistic vs Ada-ef's CLT-predicted Normal).
  - rho advantage:       Spearman rho of each method's calibration score vs.
                         the true per-query min ef -- computed from the
                         calibration pass this script already runs, no extra
                         searches. Advantage = |rho ours, best K| - |rho Ada-ef|.
  - Spread (Factor C):   P90-calibration avg ef / Mean-calibration avg ef (the
                         proxy used throughout summary.md §7), plus the direct
                         distribution of calib_min_ef.

Outcome is reported at EQUAL QUALITY, not per chosen config: the DC our
cost/quality frontier (every K x recipe) needs to reach Ada-ef's mean recall
and Ada-ef's target-hit rate, linearly interpolated between neighbouring
frontier points -- same method as the Eight-Dataset Results page.

Ada-ef is calibrated on the SAME calibration queries as ours (same easy/hard
mix), so neither method gets a calibration-distribution advantage; the
self-sampled-calibration control is deliberately left out (Factor B is not
what these experiments test).

Usage:
  python controlled_datasets.py                  # once: generate data
  python benchmark_controlled.py --config A_sift_hard30
  python summarize_controlled.py                 # after all configs: table + plots
"""

import os, sys, time, pickle, json, argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Controlled-factor Ada-ef comparison")
parser.add_argument("--config", required=True)
args = parser.parse_args()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results_controlled_{args.config}_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')

class Logger(object):
    def __init__(self, filename=os.path.join(RESULTS_DIR, "benchmark_controlled.log")):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()          # stdout is a pipe under tee/nohup, which Python block-buffers
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp
from benchmark_skewed import compute_ground_truth
from diagnose_anisotropy import compute_mean_cov, anisotropy_metrics, normality_check
from controlled_datasets import CONFIGS, load_config

np.random.seed(42)

if args.config not in CONFIGS:
    sys.exit(f"Unknown config {args.config!r}. Choose from: {', '.join(CONFIGS)}")

# ═══════════════════════════════════════════════════════════════════════
#  Configuration -- K=100 group, identical to benchmark_sift128.py
# ═══════════════════════════════════════════════════════════════════════
K_SEARCH       = 100
TARGET_RECALL  = 0.95
EF_SWEEP       = list(range(50, 3001, 50))
PROBE_COUNT    = 100
NUM_BINS       = 5
QUANTILE_STEP  = 1e-3
STATICS_LENGTH = 1025
K_SWEEP        = [1, 50, 100, 200]
KS_N_QUERIES   = 30       # same as diagnose_anisotropy.py
KS_N_CORPUS    = 20000

BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]

# ═══════════════════════════════════════════════════════════════════════
#  Helpers (unchanged from benchmark_sift128.py)
# ═══════════════════════════════════════════════════════════════════════
def build_ef_table_pct(scores_int, required_efs, pct):
    return {int(s): int(np.percentile(required_efs[scores_int == s], pct)) for s in np.unique(scores_int)}

def build_ef_table_mean(scores_int, required_efs):
    return {int(s): int(np.mean(required_efs[scores_int == s])) for s in np.unique(scores_int)}

def build_isotonic_ef_table(scores_int, required_efs, min_ef, max_ef):
    iso = IsotonicRegression(increasing='auto', out_of_bounds='clip')
    iso.fit(scores_int, required_efs)
    max_score = int(scores_int.max()) if len(scores_int) else 0
    predicted = iso.predict(np.arange(max_score + 1))
    return [int(np.clip(v, min_ef, max_ef)) for v in predicted]

def lookup_ef(score, table, min_ef=K_SEARCH, max_ef=EF_SWEEP[-1]):
    if not table: return max_ef
    if score in table: return int(np.clip(table[score], min_ef, max_ef))
    known = sorted(table.keys())
    if score <= known[0]: return int(np.clip(table[known[0]], min_ef, max_ef))
    if score >= known[-1]: return int(np.clip(table[known[-1]], min_ef, max_ef))
    lo = max(k for k in known if k <= score)
    hi = min(k for k in known if k >= score)
    if lo == hi: return int(np.clip(table[lo], min_ef, max_ef))
    frac = (score - lo) / (hi - lo)
    return int(np.clip(table[lo] + frac * (table[hi] - table[lo]), min_ef, max_ef))

def table_to_list(table):
    if not table: return [K_SEARCH]
    return [lookup_ef(s, table) for s in range(max(table.keys()) + 1)]

def build_ef_table_target_recall(scores_int, calib_queries, calib_gt_arr):
    table = {}
    wae_sum = 0
    total = 0
    for s in np.unique(scores_int):
        bucket_mask = (scores_int == s)
        bucket_queries = calib_queries[bucket_mask]
        bucket_gt = calib_gt_arr[bucket_mask]
        n_bucket = len(bucket_queries)

        bucket_ef = EF_SWEEP[-1]
        for ef in EF_SWEEP:
            bucket_recs = []
            for i in range(n_bucket):
                labs, _ = idx.search_knn_adaptive(bucket_queries[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
                bucket_recs.append(len(set(labs) & set(bucket_gt[i])) / K_SEARCH)
            if np.mean(bucket_recs) >= TARGET_RECALL:
                bucket_ef = ef
                break

        table[int(s)] = int(bucket_ef)
        wae_sum += n_bucket * bucket_ef
        total += n_bucket

    wae = int(wae_sum / total) if total > 0 else EF_SWEEP[-1]
    return table, wae

def load_or_build_target_recall_table(cache_path, scores_int, calib_queries, calib_gt_arr):
    if os.path.exists(cache_path):
        with open(cache_path) as f_cache:
            cached = json.load(f_cache)
        return {int(k): v for k, v in cached["table"].items()}, cached["wae"]
    table, wae = build_ef_table_target_recall(scores_int, calib_queries, calib_gt_arr)
    with open(cache_path, "w") as f_cache:
        json.dump({"table": table, "wae": wae}, f_cache, indent=4)
    return table, wae

def cluster_centroid_sqdists(corpus, labels, k, centroid, chunk=300_000):
    centroid = centroid.reshape(1, -1)
    parts = []
    n = corpus.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        mask = labels[start:end] == k
        if not mask.any():
            continue
        sub = corpus[start:end][mask]
        parts.append(cdist(sub, centroid, metric='sqeuclidean').flatten())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

def dc_at_quality(results, metric, goal):
    """DC our lower-left frontier (every 'Ours' row) needs to reach `goal` on
    `metric`, interpolated between neighbouring frontier points.
    bound=True: our cheapest frontier point already beats `goal`, so the true
    equal-quality cost is at most this."""
    pts = sorted((r for r in results if r['name'].startswith('Ours')),
                 key=lambda r: (r['hnsw_dc'] + r['probe_dc'], -r[metric]))
    front, best = [], -np.inf
    for p in pts:
        if p[metric] > best:
            front.append(p)
            best = p[metric]
    tot = lambda r: r['hnsw_dc'] + r['probe_dc']
    if front[0][metric] >= goal:
        return dict(dc=tot(front[0]), bound=True, via=[front[0]['name']])
    for a, b in zip(front, front[1:]):
        if b[metric] >= goal:
            t = (goal - a[metric]) / (b[metric] - a[metric])
            return dict(dc=tot(a) + t * (tot(b) - tot(a)), bound=False, via=[a['name'], b['name']])
    return None

# ═══════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════
print("═" * 80)
print(f"  Controlled experiment: {args.config}")
print("═" * 80)
corpus, calib_q, test_q, is_hard_calib, is_hard_test, meta = load_config(args.config)
ckey = meta['corpus_key']
CACHE_TAG = meta['cache_tag']   # config name, plus generator settings for synthetic configs
dim = corpus.shape[1]
n_corpus = corpus.shape[0]
N_CALIB = len(calib_q)
n_test = len(test_q)
print(f"  Experiment {meta['experiment']} | base={meta['base']} alpha={meta['alpha']} "
      f"hard_frac={meta['hard_frac']} (sigma={meta['hard_sigma']})")
print(f"  Corpus: {corpus.shape} | Calib Q: {calib_q.shape} ({is_hard_calib.mean()*100:.0f}% hard) "
      f"| Test Q: {test_q.shape} ({is_hard_test.mean()*100:.0f}% hard)")
_norms = np.linalg.norm(corpus[:1000], axis=1)
print(f"  Corpus vector norm check (first 1000): mean={_norms.mean():.4f}, std={_norms.std():.4f}")

gt_path = f"ground_truth_ctrl_{CACHE_TAG}_k{K_SEARCH}.npz"
if os.path.exists(gt_path):
    print(f"\nLoading ground truth from cache {gt_path}...")
    gt_data = np.load(gt_path)
    calib_gt, test_gt = gt_data['calib_gt'], gt_data['test_gt']
else:
    print(f"\nComputing ground truth (topk={K_SEARCH})...")
    t0 = time.time()
    calib_gt = compute_ground_truth(corpus, calib_q, k=K_SEARCH)
    test_gt = compute_ground_truth(corpus, test_q, k=K_SEARCH)
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(gt_path, calib_gt=calib_gt, test_gt=test_gt)

# ═══════════════════════════════════════════════════════════════════════
#  Factor A: KS fit of the score distribution vs Ada-ef's CLT Normal
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  Factor A: KS fit ({KS_N_QUERIES} test queries x {KS_N_CORPUS} corpus points)")
print(f"{'═' * 80}")
t0 = time.time()
mean_v, cov_v = compute_mean_cov(corpus)
aniso = anisotropy_metrics(cov_v)
ks_per_query, ks_summary = normality_check(corpus, mean_v, cov_v, test_q, RESULTS_DIR,
                                           KS_N_QUERIES, KS_N_CORPUS, n_qqplot=0)
ks_mean = float(np.mean([r['ks_statistic'] for r in ks_per_query]))
print(f"  Mean KS statistic: {ks_mean:.4f} | top-1 eigenvector share {aniso['top1_variance_frac']*100:.1f}% "
      f"| participation ratio {aniso['participation_ratio_frac_of_dim']*100:.1f}% of dim "
      f"({time.time() - t0:.1f}s)")
del cov_v

# ═══════════════════════════════════════════════════════════════════════
#  HNSW Index (M=16, ef_construction=500) -- shared by every config on this corpus
# ═══════════════════════════════════════════════════════════════════════
index_path = f"{ckey}_efc500_m16.index"   # sift128 reuses benchmark_sift128.py's index
idx = chao_hybrid_ada_ef_cpp.Index(space='l2', dim=dim)
if os.path.exists(index_path):
    print(f"\nLoading HNSW index from {index_path}...")
    idx.load_index(index_path, max_elements=n_corpus)
else:
    print("\nBuilding HNSW index...")
    t0 = time.time()
    idx.init_index(max_elements=n_corpus, ef_construction=500, M=16)
    idx.add_items(corpus)
    idx.save_index(index_path)
    print(f"  Done in {time.time() - t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
#  Shared Calibration (per-query min ef) -- also the direct spread measurement
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  Shared Calibration (Individual Query Min-EF)")
print(f"{'═' * 80}")
minef_cache_path = f"ctrl_{CACHE_TAG}_calib_min_ef_{N_CALIB}q_k{K_SEARCH}.npz"
if os.path.exists(minef_cache_path):
    print(f"  Loading calib_min_ef from cache {minef_cache_path}...")
    _c = np.load(minef_cache_path)
    calib_min_ef, calib_capped = _c['calib_min_ef'], _c['calib_capped']
else:
    t0 = time.time()
    calib_min_ef = np.zeros(N_CALIB, dtype=np.float32)
    calib_capped = np.zeros(N_CALIB, dtype=bool)
    for i in range(N_CALIB):
        for ef in EF_SWEEP:
            labs, _ = idx.search_knn_adaptive(calib_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
            if len(set(labs) & set(calib_gt[i])) / K_SEARCH >= TARGET_RECALL:
                calib_min_ef[i] = ef
                break
        else:
            calib_min_ef[i] = EF_SWEEP[-1]
            calib_capped[i] = True
        if (i + 1) % 500 == 0:
            print(f"  ... calibrated {i + 1} queries")
    print(f"  Done in {time.time() - t0:.1f}s")
    np.savez(minef_cache_path, calib_min_ef=calib_min_ef, calib_capped=calib_capped)

minef_stats = dict(
    mean=float(calib_min_ef.mean()), median=float(np.median(calib_min_ef)),
    p10=float(np.percentile(calib_min_ef, 10)), p90=float(np.percentile(calib_min_ef, 90)),
    cv=float(calib_min_ef.std() / calib_min_ef.mean()),
    p90_over_median=float(np.percentile(calib_min_ef, 90) / np.median(calib_min_ef)),
    frac_capped_at_max_ef=float(calib_capped.mean()),
    frac_at_ef_floor=float(np.mean(calib_min_ef <= K_SEARCH)),   # ef<K behaves as ef=K: no method can go cheaper
    mean_easy=float(calib_min_ef[~is_hard_calib].mean()) if (~is_hard_calib).any() else None,
    mean_hard=float(calib_min_ef[is_hard_calib].mean()) if is_hard_calib.any() else None,
)
print(f"  calib_min_ef: mean={minef_stats['mean']:.0f} median={minef_stats['median']:.0f} "
      f"P10={minef_stats['p10']:.0f} P90={minef_stats['p90']:.0f} CV={minef_stats['cv']:.2f} "
      f"capped={minef_stats['frac_capped_at_max_ef']*100:.1f}% at-floor(ef<=K)={minef_stats['frac_at_ef_floor']*100:.1f}%")
if minef_stats['mean_hard'] is not None:
    print(f"  easy queries mean min-ef={minef_stats['mean_easy']:.0f} | hard queries mean min-ef="
          f"{minef_stats['mean_hard']:.0f}  (hard should be clearly larger, or the spread knob isn't working)")
if minef_stats['frac_capped_at_max_ef'] > 0.02:
    print(f"  WARNING: {minef_stats['frac_capped_at_max_ef']*100:.1f}% of calibration queries never reached "
          f"target recall by ef={EF_SWEEP[-1]} -- consider lowering HARD_SIGMA in controlled_datasets.py.")

# ═══════════════════════════════════════════════════════════════════════
#  ADA-EF Offline Phase (paper-exact, same calibration queries as ours)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ADA-EF: Exact Paper Offline Phase")
print(f"{'═' * 80}")
t0 = time.time()
ada_scorer = chao_hybrid_ada_ef_cpp.AdaEfPaperScorer(corpus, QUANTILE_STEP)
ada_calib_scores = np.array([
    idx.adaptive_search_knn_paper(calib_q[i], K_SEARCH, STATICS_LENGTH, ada_scorer, None)[2]
    for i in range(N_CALIB)
], dtype=np.float64)
ada_scores_int = np.round(ada_calib_scores).astype(int)
rho_ada, p_ada = spearmanr(ada_calib_scores, calib_min_ef)
print(f"  Ada-ef score vs true min-EF: Spearman rho = {rho_ada:.4f} (p={p_ada:.2e})")

ada_table_exact, WAE = load_or_build_target_recall_table(
    f"cache_target_recall_ada_paper_ctrl_{CACHE_TAG}.json", ada_scores_int, calib_q, calib_gt)
print(f"  Calculated WAE for Ada-ef: {WAE} ({time.time() - t0:.1f}s)")
with open(os.path.join(RESULTS_DIR, "ef_table_ada_exact.json"), "w") as f_json:
    json.dump(ada_table_exact, f_json, indent=4)
ef_recall_estimators = [(int(s), [(int(ef), float(TARGET_RECALL))]) for s, ef in ada_table_exact.items()]
ada_sketch = chao_hybrid_ada_ef_cpp.AdaEfPaperSketch(ef_recall_estimators, TARGET_RECALL)

# ═══════════════════════════════════════════════════════════════════════
#  ONLINE EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════
all_results = []

def finish(name, recs, dc, probe_dc, dt, avg_ef, **extra):
    r = np.array(recs)
    out = dict(name=name, mean_r=float(np.mean(r)), p5=float(np.percentile(r, 5)), p1=float(np.percentile(r, 1)),
               hnsw_dc=float(dc), probe_dc=probe_dc, time=dt, avg_ef=float(avg_ef),
               pct_target=float(np.mean(r >= TARGET_RECALL) * 100))
    if is_hard_test.any():
        out['mean_r_easy'] = float(r[~is_hard_test].mean())
        out['mean_r_hard'] = float(r[is_hard_test].mean())
    out.update(extra)
    return out

def eval_vanilla(name, ef):
    idx.reset_dist_count()
    recs = []
    t0 = time.time()
    for i in range(n_test):
        labs, _ = idx.search_knn_adaptive(test_q[i], K_SEARCH, idx.entry_point, idx.max_level, ef)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    return finish(name, recs, idx.get_dist_count() / n_test, 0, time.time() - t0, ef)

def eval_ada_ef(name, scorer, sketch):
    idx.reset_dist_count()
    recs, efs = [], []
    t0 = time.time()
    for i in range(n_test):
        labs, _, _score, ef_used = idx.adaptive_search_knn_paper(test_q[i], K_SEARCH, STATICS_LENGTH, scorer, sketch)
        efs.append(ef_used)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    return finish(name, recs, idx.get_dist_count() / n_test, 0, time.time() - t0, np.mean(efs))

def eval_cluster_aware(name, K_VAL, centroids, cluster_bins, ef_table_list):
    test_nearest = np.argmin(cdist(test_q, centroids, metric='sqeuclidean'), axis=1)
    idx.reset_dist_count()
    recs, efs = [], []
    t0 = time.time()
    for i in range(n_test):
        bins = cluster_bins[test_nearest[i]].tolist()
        labs, _, ef_used = idx.search_knn_dynamic_weighted(
            test_q[i], K_SEARCH, bins, BIN_WEIGHTS, ef_table_list,
            K_SEARCH, EF_SWEEP[-1], PROBE_COUNT)
        efs.append(ef_used)
        recs.append(len(set(labs) & set(test_gt[i])) / K_SEARCH)
    return finish(name, recs, idx.get_dist_count() / n_test, K_VAL, time.time() - t0, np.mean(efs))

# ═══════════════════════════════════════════════════════════════════════
#  RUN EVALUATIONS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  ONLINE EVALUATION  (Recall@{K_SEARCH}, target={TARGET_RECALL}, n_test={n_test})")
print(f"{'═' * 80}")

for ef in [100, 200, 400, 800, 1600]:
    print(f"  Vanilla(ef={ef})...", end=" ", flush=True)
    r = eval_vanilla(f"Vanilla(ef={ef})", ef)
    print(f"R={r['mean_r']:.4f}")
    all_results.append(r)

print(f"  Ada-ef (exact)...", end=" ", flush=True)
ada_row = eval_ada_ef('Ada-ef (exact)', ada_scorer, ada_sketch)
print(f"R={ada_row['mean_r']:.4f}")
all_results.append(ada_row)

rho_ours = {}
spread_proxy = {}
for K_CLUSTERS in K_SWEEP:
    print(f"\n{'─' * 80}")
    print(f"  Cluster-Aware with K={K_CLUSTERS}")
    print(f"{'─' * 80}")
    cache_file = f"kmeans_cache_k{K_CLUSTERS}_{ckey}.pkl"   # sift128 reuses benchmark_sift128.py's cache
    if os.path.exists(cache_file):
        print(f"  [CACHE] Loading K-Means model and bins from {cache_file}...")
        with open(cache_file, 'rb') as f_cache:
            km, centroids, labels, cluster_bins = pickle.load(f_cache)
    else:
        print(f"  [COMPUTE] Running K-Means and computing bins for K={K_CLUSTERS}...")
        km = MiniBatchKMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=3, batch_size=4096)
        km.fit(corpus)
        centroids = km.cluster_centers_.astype(np.float32)
        labels = km.labels_
        CLUSTER_PCTS = [QUANTILE_STEP * (i + 1) * 100 for i in range(NUM_BINS)]
        cluster_bins = np.zeros((K_CLUSTERS, NUM_BINS), dtype=np.float32)
        for k in range(K_CLUSTERS):
            dists = cluster_centroid_sqdists(corpus, labels, k, centroids[k])
            if len(dists) > 0:
                cluster_bins[k] = np.percentile(dists, CLUSTER_PCTS)
            else:
                cluster_bins[k] = np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=np.float32)
        with open(cache_file, 'wb') as f_cache:
            pickle.dump((km, centroids, labels, cluster_bins), f_cache)

    calib_nearest = np.argmin(cdist(calib_q, centroids, metric='sqeuclidean'), axis=1)
    clust_calib_scores = np.array([
        idx.get_dynamic_probe_score_weighted(calib_q[i], cluster_bins[calib_nearest[i]].tolist(),
                                             BIN_WEIGHTS, PROBE_COUNT)
        for i in range(N_CALIB)
    ], dtype=np.float32)
    rho_k, _ = spearmanr(clust_calib_scores, calib_min_ef)
    rho_ours[K_CLUSTERS] = float(rho_k)
    print(f"  Cluster-aware (K={K_CLUSTERS}) score vs true min-EF: Spearman rho = {rho_k:.4f}")
    clust_calib_int = np.round(clust_calib_scores).astype(int)

    recipes = [
        ("Isotonic", build_isotonic_ef_table(clust_calib_int, calib_min_ef, K_SEARCH, EF_SWEEP[-1])),
        ("Mean", table_to_list(build_ef_table_mean(clust_calib_int, calib_min_ef))),
        ("P90", table_to_list(build_ef_table_pct(clust_calib_int, calib_min_ef, 90))),
        ("P70", table_to_list(build_ef_table_pct(clust_calib_int, calib_min_ef, 70))),
    ]
    by_recipe = {}
    for recipe, ef_list in recipes:
        with open(os.path.join(RESULTS_DIR, f"ef_table_k{K_CLUSTERS}_{recipe.lower()}.json"), "w") as f_json:
            json.dump(ef_list, f_json)
        print(f"  Running Online Evaluation ({recipe})...", end=" ", flush=True)
        r = eval_cluster_aware(f"Ours (K={K_CLUSTERS}, {recipe})", K_CLUSTERS, centroids, cluster_bins, ef_list)
        print(f"R={r['mean_r']:.4f}")
        all_results.append(r)
        by_recipe[recipe] = r
    spread_proxy[K_CLUSTERS] = by_recipe["P90"]['avg_ef'] / by_recipe["Mean"]['avg_ef']

# ═══════════════════════════════════════════════════════════════════════
#  Final Results + factor summary
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 80}")
print(f"  FINAL RESULTS -- {args.config} (target recall = {TARGET_RECALL}, n_test={n_test})")
print(f"{'═' * 80}\n")
hard_cols = is_hard_test.any()
hdr = (f"{'Method':<28} {'Mean R':>7} {'5th%':>7} {'HNSW DC':>8} {'+Probe':>7} {'=Total':>8} "
       f"{'Avg EF':>7} {'>=tgt%':>7}" + (f" {'R easy':>7} {'R hard':>7}" if hard_cols else ""))
print(hdr)
print("─" * len(hdr))
for r in all_results:
    probe = f"+{r['probe_dc']}" if r['probe_dc'] > 0 else ""
    line = (f"{r['name']:<28} {r['mean_r']:>7.4f} {r['p5']:>7.4f} {r['hnsw_dc']:>8.0f} {probe:>7} "
            f"{r['hnsw_dc'] + r['probe_dc']:>8.0f} {r['avg_ef']:>7.1f} {r['pct_target']:>6.1f}%")
    if hard_cols:
        line += f" {r['mean_r_easy']:>7.4f} {r['mean_r_hard']:>7.4f}"
    print(line)

ada_dc = ada_row['hnsw_dc'] + ada_row['probe_dc']
at_recall = dc_at_quality(all_results, 'mean_r', ada_row['mean_r'])
at_target = dc_at_quality(all_results, 'pct_target', ada_row['pct_target'])
best_k = max(rho_ours, key=lambda k: abs(rho_ours[k]))
summary = dict(
    config=args.config, **meta,
    n_calib=N_CALIB, n_test=n_test,
    ks_mean=ks_mean, anisotropy=aniso, ks_summary=ks_summary,
    rho_ada=float(rho_ada), rho_ours_by_k={str(k): v for k, v in rho_ours.items()},
    rho_ours_best=rho_ours[best_k], rho_ours_best_k=best_k,
    rho_advantage=abs(rho_ours[best_k]) - abs(rho_ada),
    spread_proxy_by_k={str(k): v for k, v in spread_proxy.items()},
    spread_proxy_k1=spread_proxy[1],
    calib_min_ef=minef_stats,
    ada=dict(dc=ada_dc, recall=ada_row['mean_r'], target=ada_row['pct_target']),
    equal_recall=None if at_recall is None else dict(at_recall, delta_pct=(at_recall['dc'] - ada_dc) / ada_dc * 100),
    equal_target=None if at_target is None else dict(at_target, delta_pct=(at_target['dc'] - ada_dc) / ada_dc * 100),
)

print(f"\n{'═' * 80}")
print(f"  FACTOR SUMMARY -- {args.config}")
print(f"{'═' * 80}")
print(f"  KS (Factor A):        {ks_mean:.4f}")
print(f"  rho Ada-ef:           {rho_ada:+.4f}")
print(f"  rho ours (best K={best_k}): {rho_ours[best_k]:+.4f}   advantage {summary['rho_advantage']:+.4f}")
print(f"  Spread (P90/Mean, K=1): {spread_proxy[1]:.2f}x   | calib_min_ef P90/median {minef_stats['p90_over_median']:.2f}x, CV {minef_stats['cv']:.2f}")
for label, res in (("recall", summary['equal_recall']), ("target-hit", summary['equal_target'])):
    if res is None:
        print(f"  DC at Ada-ef {label}: never reached by our frontier")
    else:
        le = "<= " if res['bound'] else ""
        print(f"  DC at Ada-ef {label}: {le}{res['dc']:.0f} vs {ada_dc:.0f}  ({le}{res['delta_pct']:+.1f}%)  via {' -> '.join(res['via'])}")

with open(os.path.join(RESULTS_DIR, "exact_sweep_results.json"), "w") as f:
    json.dump(all_results, f, indent=4)
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWrote {RESULTS_DIR}/summary.json")
