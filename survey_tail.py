#!/usr/bin/env python3
"""
Tail-weighted alternatives to the KS offline test (updateAsOf290926.md §3.1).

Why: KS (largest gap between the empirical CDF of s = q.v and Ada-ef's CLT Normal(q.mean, q'Cov q))
is dominated by the bulk of the distribution. Ada-ef's score is driven by the upper tail, where the
query's nearest corpus points lie. BIGANN (SIFT-1B) passes KS (0.029, "Ada-ef side") but ours ranks
queries far better there (rho 0.74 vs 0.42): a Gaussian bulk with a non-Gaussian upper tail.

For every benchmark dataset, on the raw vectors only (no index, same sampling as survey_ks_vibe.py:
<= 1M corpus rows / 1 GB for mean and covariance, 200 test queries, unit-normalized), per query:
  ks          KS statistic on 20,000 corpus points (as before, for reference)
  ad          Anderson-Darling A^2 / n against the same Normal on those points (weights the tails)
  tailz_p     (empirical (1-p)-quantile of s - mu) / sd  -  z_(1-p),   p = 1e-2, 1e-3, 1e-4,
              on the whole corpus sample: how far the true upper tail sits from the Normal's, in sd
              (0 = Gaussian tail; < 0 lighter, > 0 heavier)
  exc3        log10( share of the sample above the Normal's 99.9% point / 0.001 )
Each statistic is averaged over the queries (mean of |.| as well as signed for tailz, median for ad).

Then each statistic is compared with what the benchmark observed on the same dataset: the margin
m = mean over settings P, R of (|rho_ours(K=1)| - |rho_Ada-ef|) from summary_{P,R}.json of the latest
results_unified_<dataset>_* run (positive: our score ranks better). Reported per statistic: Spearman
with m, and how well a single threshold separates "ours" from "Ada-ef" datasets (best threshold on
all, and leave-one-out), and which side BIGANN falls on. With ~18 labelled datasets and several
candidates, a winner here is a hypothesis to pre-register and test on new datasets, not a result.

Usage (on the server, from the repo root, where the data and results_unified_* folders are):
  python3 survey_tail.py                         # every benchmark dataset whose files are present
  python3 survey_tail.py --datasets bigann,sift128,glove100
  python3 survey_tail.py --results-dir server_results   # labels from another folder
"""

import os, sys, glob, gzip, json, time, argparse
from datetime import datetime

import numpy as np
from scipy.stats import norm, kstest, spearmanr

import survey_ks_vibe as sv
from survey_ks_standard import bin_loader

P_TAIL = (1e-2, 1e-3, 1e-4)
CHUNK = 200_000
SD = "standard_data"


class ConcatRows:
    """Several on-disk arrays read as one (sample_rows takes contiguous slices)."""
    def __init__(self, parts):
        self.parts = parts
        self.off = np.cumsum([0] + [p.shape[0] for p in parts])
        self.shape = (int(self.off[-1]), int(parts[0].shape[1]))
        self.dtype = parts[0].dtype

    def __getitem__(self, sl):
        a = sl.start or 0
        b = self.shape[0] if sl.stop is None else min(sl.stop, self.shape[0])
        out = []
        for i, p in enumerate(self.parts):
            lo, hi = self.off[i], self.off[i + 1]
            s, e = max(a, lo), min(b, hi)
            if s < e:
                out.append(np.asarray(p[s - lo:e - lo]))
        return np.concatenate(out)


def cohere_loader(n_files=5):
    def load():
        d = "cohere_msmarco_v21_npy"
        files = sorted(glob.glob(os.path.join(d, "msmarco_v2.1_doc_segmented_*.npy")))[:n_files]
        with gzip.open(os.path.join(d, "queries.jsonl.gz"), "rt", encoding="utf-8") as fq:
            q = np.array([json.loads(line)["emb"] for line in fq], dtype=np.float32)
        return ConcatRows([np.load(f, mmap_mode="r") for f in files]), q, {}, (lambda: None)
    return load


def laion_loader(n_shards=20):
    """LAION-I2I: queries are corpus images (the benchmark holds them out of the index)."""
    def load():
        parts = [np.load(os.path.join("laion_i2i_subset", "shards", f"img_emb_{i}.npy"), mmap_mode="r")
                 for i in range(n_shards)]
        c = ConcatRows(parts)
        ids = np.sort(np.random.default_rng(7).choice(c.shape[0], 1000, replace=False))
        q = np.stack([c[slice(int(i), int(i) + 1)][0] for i in ids])
        return c, q, {}, (lambda: None)
    return load


def big_ann(name, dtype, dim, rows=2_000_000):
    b, q = os.path.join(SD, f"{name}_base_{rows}.bin"), os.path.join(SD, f"{name}_query.bin")
    return [b, q], bin_loader(b, q, dtype, dim, rows), True


def h5(path, *a):
    return [path], sv.h5_loader(path, *a), False


def sources():
    """Benchmark dataset -> (files that must exist, loader, integer vectors allowed)."""
    local = sv.local_sources()
    src = {k: ([f], loader, False) for k, (f, loader, _) in local.items()}
    src.update({
        "cohere1024": ([os.path.join("cohere_msmarco_v21_npy", "queries.jsonl.gz")], cohere_loader(), False),
        "laion_i2i": ([os.path.join("laion_i2i_subset", "shards", "img_emb_19.npy")], laion_loader(), False),
        "vibe_landmark_dino": h5(os.path.join("vibe_data", "landmark-dino-768-cosine.hdf5")),
        "vibe_inaturalist_resnet": h5(os.path.join("vibe_data", "inaturalist-resnet-2048-cosine.hdf5")),
        "vibe_yahoo_minilm": h5(os.path.join("vibe_data", "yahoo-minilm-384-normalized.hdf5")),
        "vibe_imagenet_align": h5(os.path.join("vibe_data", "imagenet-align-640-normalized.hdf5")),
        "gist960": h5(os.path.join(SD, "gist-960-euclidean.hdf5")),
        "fashionmnist784": h5(os.path.join(SD, "fashion-mnist-784-euclidean.hdf5")),
        "lastfm64": h5(os.path.join(SD, "lastfm-64-dot.hdf5")),
        "coco_i2i": h5(os.path.join(SD, "coco-i2i-512-angular.hdf5")),
        "coco_t2i": h5(os.path.join(SD, "coco-t2i-512-angular.hdf5")),
        "deep1b": big_ann("deep1b", "float32", 96),
        "bigann": big_ann("bigann", "uint8", 128),
        "msturing": big_ann("msturing", "float32", 100),
    })
    return src


def ad_per_n(s, mu, sd):
    """Anderson-Darling A^2 / n against the fully specified Normal(mu, sd)."""
    z = np.sort((s - mu) / sd)
    n = len(z)
    i = np.arange(1, n + 1)
    return float((-n - np.mean((2 * i - 1) * (norm.logcdf(z) + norm.logsf(z)[::-1]))) / n)


def tail_stats(corpus, queries, rng):
    """corpus, queries: unit-normalized float32. Per-query statistics, then summaries."""
    x = corpus.astype(np.float64)
    mean, cov = x.mean(axis=0), np.cov(x, rowvar=False)
    del x
    qi = rng.choice(len(queries), min(sv.N_QUERIES, len(queries)), replace=False)
    Q = queries[qi].astype(np.float64)
    mu = Q @ mean
    sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Q, cov, Q), 1e-18))
    sims = np.empty((len(corpus), len(Q)), dtype=np.float32)          # all sample sims, per query
    for a in range(0, len(corpus), CHUNK):
        sims[a:a + CHUNK] = corpus[a:a + CHUNK] @ Q.T.astype(np.float32)
    sub = rng.choice(len(corpus), min(sv.N_SAMPLE, len(corpus)), replace=False)
    per = {k: [] for k in ["ks", "ad", "exc3"] + [f"tailz_{p:g}" for p in P_TAIL]}
    qs = np.quantile(sims, [1 - p for p in P_TAIL], axis=0)             # (len(P_TAIL), n_q)
    for j in range(len(Q)):
        s_sub = sims[sub, j].astype(np.float64)
        per["ks"].append(kstest(s_sub, "norm", args=(mu[j], sd[j])).statistic)
        per["ad"].append(ad_per_n(s_sub, mu[j], sd[j]))
        for k, p in enumerate(P_TAIL):
            per[f"tailz_{p:g}"].append((qs[k, j] - mu[j]) / sd[j] - norm.isf(p))
        thr = mu[j] + sd[j] * norm.isf(1e-3)
        frac = max(np.mean(sims[:, j] > thr), 0.5 / len(corpus))
        per["exc3"].append(np.log10(frac / 1e-3))
    out = dict(n_sample=len(corpus), n_queries=len(Q))
    for k, v in per.items():
        v = np.asarray(v)
        out[k] = float(np.median(v) if k == "ad" else v.mean())
        if k.startswith("tailz") or k == "exc3":
            out[k + "_abs"] = float(np.abs(v).mean())
    return out


def labels(results_dir):
    """dataset -> margin m (mean over P, R of |rho_ours K=1| - |rho_Ada|), None if undefined."""
    latest = {}
    for d in glob.glob(os.path.join(results_dir, "results_unified_*")):
        b = os.path.basename(d)
        if "smoke" in b or "ablation" in b:
            continue
        ds = b[len("results_unified_"):-len("_YYYYmmdd_HHMMSS")]
        if ds not in latest or b > os.path.basename(latest[ds]):
            latest[ds] = d
    out = {}
    for ds, d in latest.items():
        ms = []
        for S in ("P", "R"):
            f = os.path.join(d, f"summary_{S}.json")
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                s = json.loads(fh.read().replace("NaN", "null"))
            ro, ra = (s.get("rho_ours") or {}).get("1"), s.get("rho_ada")
            if ro is not None and ra is not None:
                ms.append(abs(ro) - abs(ra))
        out[ds] = float(np.mean(ms)) if ms else None
    return out


def separation(xs, ys):
    """Best single threshold separating ys>0 from ys<0 on xs (either direction): (correct, threshold, dir)."""
    best = (-1, None, None)
    cand = np.unique(xs)
    cuts = np.concatenate([[cand[0] - 1], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1]])
    for c in cuts:
        for dirn in (1, -1):
            pred = (dirn * (xs - c)) > 0                # True = predicts "ours"
            ok = int(np.sum(pred == (ys > 0)))
            if ok > best[0]:
                best = (ok, float(c), dirn)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="", help="comma list (default: every benchmark dataset present)")
    ap.add_argument("--results-dir", default=".", help="where the results_unified_* folders are")
    args = ap.parse_args()
    src = sources()
    wanted = [s.strip() for s in args.datasets.split(",") if s.strip()] or list(src)
    out_dir = f"results_tail_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    lab = labels(args.results_dir)

    res = {}
    for name in wanted:
        files, loader, allow_int = src[name]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print(f"  {name}: skipped, missing {missing[0]}", flush=True)
            continue
        t0 = time.time()
        rng = np.random.default_rng(sv.SEED)
        try:
            train, test_raw, _, close = loader()
            try:
                dim = train.shape[1]
                rows = int(min(sv.STATS_ROWS, sv.STATS_BYTES // (dim * 4)))
                corpus = sv.sample_rows(train, rows, rng)
                test = np.asarray(test_raw, dtype=np.float32)
            finally:
                close()
            r = tail_stats(sv.normalize(corpus.astype(np.float32)), sv.normalize(test), rng)
        except Exception as e:                       # one bad source must not stop the survey
            print(f"  {name}: error {e!r}", flush=True)
            continue
        r["margin"] = lab.get(name)
        r["seconds"] = round(time.time() - t0, 1)
        res[name] = r
        margin = "n/a" if r["margin"] is None else f"{r['margin']:+.3f}"
        print(f"  {name:<24} KS {r['ks']:.4f}  AD/n {r['ad']:.4f}  tail z (1e-2, 1e-3, 1e-4) "
              f"{r['tailz_0.01']:+.2f} {r['tailz_0.001']:+.2f} {r['tailz_0.0001']:+.2f}  exc3 {r['exc3']:+.2f}  "
              f"margin {margin}  [{r['seconds']}s]", flush=True)
        with open(os.path.join(out_dir, "tail_survey.json"), "w") as f:
            json.dump(res, f, indent=1)

    stats = ["ks", "ad", "tailz_0.01", "tailz_0.001", "tailz_0.0001", "tailz_0.01_abs", "tailz_0.001_abs",
             "tailz_0.0001_abs", "exc3", "exc3_abs"]
    known = {k: v for k, v in res.items() if v["margin"] is not None and abs(v["margin"]) > 0.02}
    print(f"\n{len(known)} datasets with a clear ranking winner (|margin| > 0.02): "
          f"{sum(v['margin'] > 0 for v in known.values())} ours, {sum(v['margin'] < 0 for v in known.values())} Ada-ef"
          f"{'' if 'bigann' in known else '   (BIGANN not among them: run it too)'}")
    if len(known) < 4:
        return
    names = sorted(known)
    ys = np.array([known[n]["margin"] for n in names])
    allm = {k: v for k, v in res.items() if v["margin"] is not None}
    print(f"{'statistic':<18} {'Spearman vs margin':>18} {'separates':>10} {'leave-one-out':>14}  BIGANN")
    for st in stats:
        xs = np.array([known[n][st] for n in names])
        rho = spearmanr([v[st] for v in allm.values()], [v["margin"] for v in allm.values()])[0]
        ok, cut, dirn = separation(xs, ys)
        loo = 0
        for i in range(len(names)):
            m = np.arange(len(names)) != i
            _, c, d = separation(xs[m], ys[m])
            loo += int(((d * (xs[i] - c)) > 0) == (ys[i] > 0))
        big = ""
        if "bigann" in known:
            pred = (dirn * (known["bigann"][st] - cut)) > 0
            big = f"{'ours' if pred else 'Ada-ef'} ({'right' if pred == (known['bigann']['margin'] > 0) else 'wrong'})"
        print(f"{st:<18} {rho:>+18.2f} {f'{ok}/{len(names)}':>10} {f'{loo}/{len(names)}':>14}  {big}")
    print(f"\n  wrote {out_dir}/tail_survey.json")


if __name__ == "__main__":
    main()
