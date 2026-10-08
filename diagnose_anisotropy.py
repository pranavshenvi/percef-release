#!/usr/bin/env python3
"""
Diagnose whether Ada-ef's CLT/Gaussian assumption actually holds on a given
corpus, and how anisotropic the corpus embeddings are -- independent
measurements, not inferred backward from which method wins online.

Background (see summary.md): Ada-ef's bin thresholds assume the score
s = q.v (v ~ corpus) is Gaussian, with mu_q = q . mu_V and
var_q = q^T Sigma_V q (Theorem 5.2, CLT over the corpus mean vector /
covariance matrix -- exactly what CosineDistanceEstimator::get_practical_distribution
computes in distribution.h). This script checks that assumption directly:

  1. Anisotropy of the corpus: eigen-spectrum of Sigma_V -> participation
     ratio (1 = all variance in one direction, dim = perfectly isotropic)
     and fraction of variance in the top-10 eigenvectors.
  2. Normality of the actual score: for a sample of real queries, draw many
     corpus points, compute s = q.v, and KS-test the empirical sample
     against the CLT-predicted Normal(mu_q, sqrt(var_q)) -- NOT a distribution
     re-fit to the sample, the actual formula's prediction. Also saves a
     QQ-plot per sampled query.

Run once per dataset (MS MARCO 384-dim, Cohere 1024-dim) with the same file
paths as benchmark_exact_paper_sweep.py / benchmark_cohere1024_sweep.py, so
results are directly comparable to the correlation (rho) findings already in
summary.md, without re-deriving the "which method won" story from them.
"""
import os, sys, json, argparse, time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════
N_QUERY_SAMPLE   = 30      # how many real queries to run the normality test on
N_CORPUS_SAMPLE  = 20000   # how many corpus points sampled per query for s = q.v
N_QQPLOT_QUERIES = 4       # how many of those queries also get a saved QQ-plot
COV_CHUNK        = 200_000 # rows per chunk when accumulating X^T X for Sigma_V

# ═══════════════════════════════════════════════════════════════════════
#  Dataset loaders (paths match the existing benchmark scripts exactly)
# ═══════════════════════════════════════════════════════════════════════
def load_msmarco384():
    import h5py
    with h5py.File('msmarco-8.8M-minilm-384d.hdf5', 'r') as f:
        corpus = f['embeddings'][:].astype(np.float32)
    test_q = np.load('msmarco_qemb_validation.npz')['emb'].astype(np.float32)
    return corpus, test_q

def load_glove100():
    import h5py
    with h5py.File('glove-100-angular.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
        test_q = f['test'][:].astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    test_q /= np.linalg.norm(test_q, axis=1, keepdims=True)
    return corpus, test_q

def load_deepimage96():
    import h5py
    with h5py.File('deep-image-96-angular.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)  # full 9.99M, matches benchmark_deep_image_new.py
        test_q = f['test'][:].astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    test_q /= np.linalg.norm(test_q, axis=1, keepdims=True)
    return corpus, test_q

def load_laion_i2i():
    DATA_DIR = "laion_i2i_subset"
    corpus = np.load(os.path.join(DATA_DIR, "corpus_emb.npy")).astype(np.float32)
    q_data = np.load(os.path.join(DATA_DIR, "queries.npz"))
    queries = q_data['emb'].astype(np.float32)
    cn = np.linalg.norm(corpus[:2000], axis=1)
    qn = np.linalg.norm(queries[:min(200, len(queries))], axis=1)
    if abs(cn.mean() - 1) > 0.01 or abs(qn.mean() - 1) > 0.01:
        print("  Not unit-normalized -- normalizing now.")
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return corpus, queries

def load_nytimes256():
    import h5py
    with h5py.File('nytimes-256-angular.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
        test_q = f['test'][:].astype(np.float32)
    # A small number of NYTimes rows are exact zero vectors (rare all-stopword
    # docs) -- clip their norm to 1 instead of dividing by zero (which would
    # silently produce NaNs that corrupt everything downstream).
    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    test_norms = np.linalg.norm(test_q, axis=1, keepdims=True)
    corpus_norms[corpus_norms == 0] = 1.0
    test_norms[test_norms == 0] = 1.0
    corpus /= corpus_norms
    test_q /= test_norms
    return corpus, test_q

def load_sift128():
    import h5py
    with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
        test_q = f['test'][:].astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    test_q /= np.linalg.norm(test_q, axis=1, keepdims=True)
    return corpus, test_q

def load_dbpedia_openai1536():
    import h5py
    with h5py.File('dbpedia-openai-1000k-angular.hdf5', 'r') as f:
        corpus = f['train'][:].astype(np.float32)
        test_q = f['test'][:].astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    test_q /= np.linalg.norm(test_q, axis=1, keepdims=True)
    return corpus, test_q

def load_yambda_audio():
    corpus = np.load('yambda_audio_corpus.npy').astype(np.float32)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    norms[norms.squeeze() == 0] = 1.0
    corpus = corpus / norms
    n = corpus.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    test_q = corpus[perm[:10000]]
    return corpus, test_q

def load_cohere1024():
    DATA_DIR = "cohere_msmarco_v21_subset"
    corpus = np.load(os.path.join(DATA_DIR, "corpus_emb.npy")).astype(np.float32)
    q_data = np.load(os.path.join(DATA_DIR, "queries.npz"))
    all_q = q_data['emb'].astype(np.float32)
    cn = np.linalg.norm(corpus[:2000], axis=1)
    qn = np.linalg.norm(all_q[:min(200, len(all_q))], axis=1)
    if abs(cn.mean() - 1) > 0.01 or abs(qn.mean() - 1) > 0.01:
        print("  Not unit-normalized -- normalizing now (matches AdaEfPaperScorer's own requirement).")
        corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        all_q = all_q / np.linalg.norm(all_q, axis=1, keepdims=True)
    return corpus, all_q

DATASETS = {
    "msmarco384": load_msmarco384,
    "cohere1024": load_cohere1024,
    "glove100": load_glove100,
    "deepimage96": load_deepimage96,
    "laion_i2i": load_laion_i2i,
    "nytimes256": load_nytimes256,
    "sift128": load_sift128,
    "dbpedia_openai1536": load_dbpedia_openai1536,
    "yambda_audio": load_yambda_audio,
}

# ═══════════════════════════════════════════════════════════════════════
#  Corpus mean / covariance (matches distribution.h's compute_mean_parallel /
#  compute_covariance_matrix2 -- population mean/covariance of the corpus,
#  same quantities Ada-ef's own CosineDistanceEstimator builds internally).
# ═══════════════════════════════════════════════════════════════════════
def compute_mean_cov(corpus, chunk=COV_CHUNK):
    n, d = corpus.shape
    mean = corpus.mean(axis=0, dtype=np.float64)
    xtx = np.zeros((d, d), dtype=np.float64)
    for start in range(0, n, chunk):
        block = corpus[start:start + chunk].astype(np.float64)
        xtx += block.T @ block
    cov = xtx / n - np.outer(mean, mean)
    return mean.astype(np.float32), cov.astype(np.float64)

def anisotropy_metrics(cov):
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)[::-1]  # descending, clip tiny negatives from fp error
    total = eigvals.sum()
    participation_ratio = float((total ** 2) / np.sum(eigvals ** 2))
    top10_frac = float(eigvals[:10].sum() / total) if total > 0 else 0.0
    return dict(
        dim=len(eigvals),
        participation_ratio=participation_ratio,
        participation_ratio_frac_of_dim=participation_ratio / len(eigvals),
        top10_variance_frac=top10_frac,
        top1_variance_frac=float(eigvals[0] / total) if total > 0 else 0.0,
        eigvals_top20=eigvals[:20].tolist(),
    )

# ═══════════════════════════════════════════════════════════════════════
#  Normality check: for each sampled query, draw N_CORPUS_SAMPLE corpus
#  points, compute s = q.v, KS-test against the CLT-PREDICTED Normal
#  (mu_q, sqrt(var_q)) -- not a distribution fit to the sample itself.
# ═══════════════════════════════════════════════════════════════════════
def normality_check(corpus, mean, cov, queries, results_dir, n_query_sample, n_corpus_sample, n_qqplot):
    n_corpus = corpus.shape[0]
    query_idx = np.random.choice(len(queries), size=min(n_query_sample, len(queries)), replace=False)
    per_query = []
    for qi, q_i in enumerate(query_idx):
        q = queries[q_i].astype(np.float64)
        mu_q = float(q @ mean)
        var_q = float(q @ cov @ q)
        sigma_q = float(np.sqrt(max(var_q, 1e-12)))

        sample_idx = np.random.choice(n_corpus, size=n_corpus_sample, replace=False)
        v_sample = corpus[sample_idx].astype(np.float64)
        s = v_sample @ q

        ks_stat, ks_p = stats.kstest(s, 'norm', args=(mu_q, sigma_q))
        emp_mean, emp_std = float(s.mean()), float(s.std())

        per_query.append(dict(
            query_index=int(q_i), mu_q_predicted=mu_q, sigma_q_predicted=sigma_q,
            emp_mean=emp_mean, emp_std=emp_std,
            ks_statistic=float(ks_stat), ks_pvalue=float(ks_p),
        ))

        if qi < n_qqplot:
            fig, ax = plt.subplots(figsize=(5, 5))
            stats.probplot(s, dist="norm", sparams=(mu_q, sigma_q), plot=ax)
            ax.set_title(f"QQ-plot vs CLT-predicted Normal, query_idx={q_i}\n"
                         f"KS p={ks_p:.4g}, mu_q={mu_q:.4f}, sigma_q={sigma_q:.4f}")
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, f"qqplot_query{q_i}.png"), dpi=120)
            plt.close(fig)

        print(f"  query {q_i}: KS stat={ks_stat:.4f} p={ks_p:.4g} "
              f"(predicted mu={mu_q:.4f} sigma={sigma_q:.4f}, empirical mean={emp_mean:.4f} std={emp_std:.4f})")

    p_values = np.array([r['ks_pvalue'] for r in per_query])
    summary = dict(
        n_queries=len(per_query),
        n_corpus_sample_per_query=n_corpus_sample,
        median_ks_pvalue=float(np.median(p_values)),
        frac_reject_normality_at_0p05=float(np.mean(p_values < 0.05)),
        frac_reject_normality_at_0p01=float(np.mean(p_values < 0.01)),
    )
    return per_query, summary

# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--n-query-sample", type=int, default=N_QUERY_SAMPLE)
    parser.add_argument("--n-corpus-sample", type=int, default=N_CORPUS_SAMPLE)
    parser.add_argument("--n-qqplot", type=int, default=N_QQPLOT_QUERIES)
    args = parser.parse_args()

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = f"results_anisotropy_{args.dataset}_{TIMESTAMP}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sys.stdout.reconfigure(encoding='utf-8')

    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(os.path.join(RESULTS_DIR, "diagnose_anisotropy.log"))

    print("=" * 80)
    print(f"  Anisotropy / CLT-normality diagnostic -- dataset={args.dataset}")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 80)

    t0 = time.time()
    corpus, queries = DATASETS[args.dataset]()
    dim = corpus.shape[1]
    print(f"  Corpus: {corpus.shape} | Queries: {queries.shape} | dim={dim}")

    print("\n  Computing corpus mean / covariance...")
    mean, cov = compute_mean_cov(corpus)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n  Anisotropy metrics (eigen-spectrum of Sigma_V)...")
    aniso = anisotropy_metrics(cov)
    print(f"  participation_ratio={aniso['participation_ratio']:.2f} / dim={dim} "
          f"({aniso['participation_ratio_frac_of_dim']*100:.1f}% of dim)")
    print(f"  top10_variance_frac={aniso['top10_variance_frac']*100:.1f}%  "
          f"top1_variance_frac={aniso['top1_variance_frac']*100:.1f}%")

    print(f"\n  Normality check ({args.n_query_sample} queries x {args.n_corpus_sample} corpus samples each)...")
    per_query, norm_summary = normality_check(
        corpus, mean, cov, queries, RESULTS_DIR,
        args.n_query_sample, args.n_corpus_sample, args.n_qqplot)
    print(f"\n  median KS p-value: {norm_summary['median_ks_pvalue']:.4g}")
    print(f"  fraction rejecting normality at alpha=0.05: {norm_summary['frac_reject_normality_at_0p05']*100:.1f}%")
    print(f"  fraction rejecting normality at alpha=0.01: {norm_summary['frac_reject_normality_at_0p01']*100:.1f}%")

    out = dict(
        dataset=args.dataset,
        corpus_shape=list(corpus.shape),
        anisotropy=aniso,
        normality_summary=norm_summary,
        per_query=per_query,
    )
    with open(os.path.join(RESULTS_DIR, "anisotropy_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    print(f"  Wrote results to {RESULTS_DIR}/anisotropy_results.json (+ QQ-plots)")

if __name__ == "__main__":
    main()
