#!/usr/bin/env python3
"""
KS survey: how Gaussian is the query-to-corpus similarity distribution on modern vector
datasets? (updateAsOf260926.md: under the paper protocol Ada-ef's Gaussian difficulty score
ranks queries better at KS <= 0.047 and ours at KS >= 0.072. Every modern retrieval embedding
measured so far sits below that band; this checks whether that holds broadly, especially for
out-of-distribution (OOD) workloads, where queries come from a different distribution than the
corpus.)

Datasets: the VIBE benchmark (arXiv 2505.17810; huggingface.co/datasets/vector-index-bench/vibe),
11 in-distribution + 8 OOD current datasets (+ 5 deprecated with --include-deprecated), float
versions only (uint8/binary/ColBERT multi-vector variants do not apply). Each HDF5 file has
`train` (corpus), `test` (1,000 queries) and, for OOD sets, `learn` (a larger sample of queries).

KS is computed as in diagnose_anisotropy.py / benchmark_unified.py, but over 200 random test
queries (those used 30, which left the estimate noisy by ~0.005-0.008; a 95% interval is now
reported and labels near a band edge are marked uncertain): s = q.v over 20,000 random corpus
points, KS statistic against Ada-ef's CLT-predicted
Normal(q.mean, q'Cov q) with the corpus mean and covariance; the dataset's KS is the mean over
queries. Vectors are unit-normalized (cosine), as in all our benchmarks. For VIBE's
inner-product datasets the KS on raw (unnormalized) inner products is also reported, since
Ada-ef has an inner-product estimator for that case. Mean and covariance come from a uniform
sample of up to 1M rows (fewer for very wide vectors); earlier checks showed KS does not move
between 1M and 10M rows (updateAsOf160926.md 4.1).

Also reported per dataset:
  ks_self  : the same KS with corpus points used as queries. For in-distribution data it should
             match ks; for OOD data the gap shows how differently real queries see the corpus
             (and how far Ada-ef's corpus-point calibration is from real queries).
  top1_var, pr_frac : top-1 eigenvalue share and participation ratio (descriptive only).
  predicted: which difficulty score the six unified runs say should rank queries better
             (Ada-ef if ks <= 0.044, ours if ks >= 0.066, "band" in between). Provisional until
             Cohere and LAION have been checked against it.

--local recomputes KS on six of our own datasets (GloVe-100, dbpedia-1536, MS MARCO-384,
DeepImage-96, Yambda, SIFT-128) so the survey can be checked against the unified runs and the
band edges re-measured with the same precision (unified values 0.016, 0.036, 0.047, 0.072,
0.095, 0.114, from 30 queries).

Usage (on the server):
  python3 survey_ks_vibe.py --local                    # validate against the unified runs first
  python3 survey_ks_vibe.py --max-gb 20                # download + survey everything up to 20 GB
  python3 survey_ks_vibe.py                            # all current datasets (~200 GB download)
  python3 survey_ks_vibe.py --datasets laion-clip-512-normalized,yandex-200-cosine
"""

import os, sys, json, time, argparse
from datetime import datetime

import numpy as np
import requests
from scipy.stats import kstest
from tqdm import tqdm

BASE = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main"

# name -> (split, modality / model, size in GB)
VIBE = {
    # in-distribution
    "agnews-mxbai-1024-euclidean":     ("ID",  "text, MXBAI",            3.16),
    "arxiv-nomic-768-normalized":      ("ID",  "text, Nomic",            4.14),
    "dpr-jina-768-normalized":         ("ID",  "text, Jina",            64.42),
    "glove-200-cosine":                ("ID",  "word, GloVe",            0.96),
    "gooaq-distilroberta-768-normalized": ("ID", "text, DistilRoBERTa",  4.54),
    "imagenet-clip-512-normalized":    ("ID",  "image, CLIP",            2.63),
    "inaturalist-resnet-2048-cosine":  ("ID",  "image, ResNet",          4.10),
    "landmark-dino-768-cosine":        ("ID",  "image, DINO",            2.34),
    "landmark-nomic-768-normalized":   ("ID",  "image, Nomic Vision",    2.34),
    "msmarco-qwen-1024-normalized":    ("ID",  "text, Qwen",            36.22),
    "yahoo-minilm-384-normalized":     ("ID",  "text, MiniLM",           1.04),
    # out-of-distribution
    "hotpotqa-harrier-640-normalized": ("OOD", "text QA, Harrier",      13.69),
    "imagenet-align-640-normalized":   ("OOD", "text-to-image, ALIGN",   4.22),
    "laion-clip-512-normalized":       ("OOD", "text-to-image, CLIP",    4.50),
    "yandex-200-cosine":               ("OOD", "text-to-image, SE-ResNeXt", 2.00),
    "cqadupstack-lemur-2048-ip":       ("OOD", "multi-vector, LEMUR",   15.20),
    "cqadupstack-muvera-5120-ip":      ("OOD", "multi-vector, MUVERA",  37.20),
    "yi-128-ip":                       ("OOD", "attention keys, Yi-6B",  0.27),
    "llama-128-ip":                    ("OOD", "attention keys, Llama-3-8B", 0.37),
}
DEPRECATED = {
    "ccnews-nomic-768-normalized":       ("ID",  "text, Nomic",          1.53),
    "celeba-resnet-2048-cosine":         ("ID",  "image, ResNet",        1.66),
    "coco-nomic-768-normalized":         ("OOD", "text-to-image, Nomic", 1.27),
    "codesearchnet-jina-768-cosine":     ("ID",  "code, Jina",           4.23),
    "simplewiki-openai-3072-normalized": ("ID",  "text, OpenAI-3072",    3.21),
}
def local_sources():
    """Our own datasets: name -> (file that must exist, loader, unified-run KS with 30 queries)."""
    return {
        "glove100":    ("glove-100-angular.hdf5", h5_loader("glove-100-angular.hdf5"), 0.016),
        "dbpedia1536": ("dbpedia-openai-1000k-angular.hdf5", h5_loader("dbpedia-openai-1000k-angular.hdf5"), 0.036),
        "msmarco384":  ("msmarco-8.8M-minilm-384d.hdf5",
                        h5_loader("msmarco-8.8M-minilm-384d.hdf5", "embeddings", "msmarco_qemb_validation.npz"), 0.047),
        "deepimage96": ("deep-image-96-angular.hdf5", h5_loader("deep-image-96-angular.hdf5"), 0.072),
        "yambda":      ("yambda_audio_corpus.npy", yambda_loader, 0.095),
        "sift128":     ("sift-128-euclidean.hdf5", h5_loader("sift-128-euclidean.hdf5"), 0.114),
    }

# Crossover from the unified runs, edges re-measured at 200 queries (--local, 2026-09-26):
# Ada-ef's score ranked better up to MS MARCO-384 (0.0442 +/- 0.0019), ours from DeepImage-96
# (0.0656 +/- 0.0044). The 30-query values had put the band at 0.047-0.072.
BAND_LO, BAND_HI = 0.044, 0.066
SEED = 42
N_QUERIES, N_SAMPLE = 200, 20000   # 30 queries (as in the unified runs) left KS noisy by ~0.005-0.008,
                                   # as large as the band itself; --n-queries overrides
STATS_ROWS, STATS_BYTES = 1_000_000, 1e9     # sample cap for mean/covariance: peak RAM stays ~5 GB,
                                             # so the survey can run next to a 40-50 GB benchmark
BLOCK = 5000                                  # contiguous rows per random read


def download(name, data_dir):
    dest = os.path.join(data_dir, name + ".hdf5")
    if os.path.exists(dest):
        return dest
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(f"{BASE}/{name}.hdf5", stream=True, timeout=120, headers=headers) as r:
        if done and r.status_code != 206:
            done = 0
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + done
        with open(part, "ab" if done else "wb") as f, \
                tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(part, dest)
    return dest


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def sample_rows(ds, n_rows, rng):
    """Uniform-ish sample: random contiguous blocks (h5py reads blocks fast, single rows slowly)."""
    n = ds.shape[0]
    if n <= n_rows:
        return np.asarray(ds[:], dtype=np.float32)
    n_blocks = int(np.ceil(n_rows / BLOCK))
    starts = np.sort(rng.choice(n // BLOCK, size=min(n_blocks, n // BLOCK), replace=False)) * BLOCK
    return np.concatenate([np.asarray(ds[s:s + BLOCK], dtype=np.float32) for s in starts])[:n_rows]


def ks_stats(corpus_sample, queries, rng):
    """Mean KS of q.v against Normal(q.mean, q'Cov q), plus descriptive spectrum stats."""
    x = corpus_sample.astype(np.float64)
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)[::-1]
    vals = []
    for qi in rng.choice(len(queries), min(N_QUERIES, len(queries)), replace=False):
        q = queries[qi].astype(np.float64)
        mu, var = float(q @ mean), float(q @ cov @ q)
        pts = corpus_sample[rng.choice(len(corpus_sample), min(N_SAMPLE, len(corpus_sample)), replace=False)]
        s = pts.astype(np.float64) @ q
        vals.append(kstest(s, "norm", args=(mu, np.sqrt(max(var, 1e-18)))).statistic)
    return dict(ks=float(np.mean(vals)), ks_ci95=float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))),
                ks_median=float(np.median(vals)), ks_std=float(np.std(vals)), n_ks_queries=len(vals),
                top1_var=float(eig[0] / eig.sum()), pr_frac=float(eig.sum() ** 2 / (eig ** 2).sum() / len(eig)))


def predicted(ks, ci=0.0):
    """Label from the point estimate; "(uncertain)" when the 95% interval crosses a band edge."""
    label = "Ada-ef" if ks <= BAND_LO else ("ours" if ks >= BAND_HI else "band")
    crosses = any(ks - ci < edge < ks + ci for edge in (BAND_LO, BAND_HI))
    return label + (" (uncertain)" if crosses else "")


def h5_loader(path, train_key="train", test_npz=None):
    """Corpus from an HDF5 file; queries from its `test` set, or from an .npz ('emb')."""
    def load():
        import h5py
        f = h5py.File(path, "r")
        test = np.load(test_npz)["emb"] if test_npz else f["test"][:]
        attrs = {k: (v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v))
                 for k, v in f.attrs.items()}
        return f[train_key], test, attrs, f.close
    return load


def yambda_loader():
    """Yambda has no query file: queries are corpus tracks, as in the unified run."""
    mm = np.load("yambda_audio_corpus.npy", mmap_mode="r")
    q = np.sort(np.random.default_rng(7).choice(mm.shape[0], 1000, replace=False))
    return mm, np.asarray(mm[q]), {}, (lambda: None)


def survey_file(path, raw_ip, rng):
    return survey_source(h5_loader(path), raw_ip, rng)


def survey_source(loader, raw_ip, rng, allow_int=False):
    t0 = time.time()
    train, test_raw, attrs, close = loader()
    try:
        n, dim = train.shape
        if not allow_int and (test_raw.dtype.kind != "f" or train.dtype.kind != "f"):
            return dict(skipped=f"non-float vectors ({train.dtype})")
        rows = int(min(STATS_ROWS, STATS_BYTES // (dim * 4)))
        corpus = sample_rows(train, rows, rng)
        test = np.asarray(test_raw, dtype=np.float32)
    finally:
        close()
    out = dict(n_corpus=int(n), dim=int(dim), n_test=len(test), stats_rows=len(corpus), attrs=attrs)
    cn, tn = normalize(corpus), normalize(test)
    out.update(ks_stats(cn, tn, rng))
    self_q = cn[rng.choice(len(cn), min(1000, len(cn)), replace=False)]
    out["ks_self"] = ks_stats(cn, self_q, rng)["ks"]
    if raw_ip:
        out["ks_raw_ip"] = ks_stats(corpus, test, rng)["ks"]
    out["predicted"] = predicted(out["ks"], out["ks_ci95"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    global N_QUERIES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="", help="comma list of VIBE names (default: all current)")
    ap.add_argument("--include-deprecated", action="store_true")
    ap.add_argument("--max-gb", type=float, default=1e9, help="skip VIBE files larger than this (unless present)")
    ap.add_argument("--data-dir", default="vibe_data")
    ap.add_argument("--local", action="store_true", help="validate on our own datasets instead of VIBE")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--n-queries", type=int, default=N_QUERIES, help="queries per KS estimate")
    args = ap.parse_args()
    N_QUERIES = args.n_queries

    out_dir = f"results_ks_survey_{'local_' if args.local else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    if args.local:
        for name, (path, loader, ref) in local_sources().items():
            if not os.path.exists(path):
                print(f"  {name}: {path} not found, skipped"); continue
            print(f"  {name} ...", flush=True)
            r = survey_source(loader, False, np.random.default_rng(SEED))
            r.update(split="ours", model="", unified_ks=ref)
            results[name] = r
            print(f"    KS {r['ks']:.4f} ± {r['ks_ci95']:.4f} (unified run: {ref:.3f})  self {r['ks_self']:.4f}  [{r['seconds']}s]", flush=True)
    else:
        catalog = dict(VIBE, **(DEPRECATED if args.include_deprecated else {}))
        names = [s.strip() for s in args.datasets.split(",") if s.strip()] or list(catalog)
        os.makedirs(args.data_dir, exist_ok=True)
        for name in sorted(names, key=lambda n: catalog[n][2]):
            split, model, gb = catalog[name]
            present = os.path.exists(os.path.join(args.data_dir, name + ".hdf5"))
            if gb > args.max_gb and not present:
                print(f"  {name}: {gb:.1f} GB > --max-gb, skipped"); continue
            path = download(name, args.data_dir)
            if args.download_only:
                continue
            print(f"  {name} ...", flush=True)
            try:
                r = survey_file(path, name.endswith("-ip"), np.random.default_rng(SEED))
            except Exception as e:           # one bad file must not stop the survey
                r = dict(error=repr(e))
            r.update(split=split, model=model)
            results[name] = r
            if "ks" in r:
                extra = f"  raw-IP {r['ks_raw_ip']:.4f}" if "ks_raw_ip" in r else ""
                print(f"    KS {r['ks']:.4f} ± {r['ks_ci95']:.4f} (self {r['ks_self']:.4f}){extra} -> {r['predicted']}  [{r['seconds']}s]", flush=True)
            else:
                print(f"    {r}", flush=True)
            with open(os.path.join(out_dir, "ks_survey.json"), "w") as f:
                json.dump(results, f, indent=1)
        if args.download_only:
            return

    with open(os.path.join(out_dir, "ks_survey.json"), "w") as f:
        json.dump(results, f, indent=1)
    done = {k: v for k, v in results.items() if "ks" in v}
    print(f"\n{'dataset':<38} {'split':<5} {'model':<26} {'KS':>7} {'±95%':>7} {'self':>7} {'rawIP':>7} {'top1%':>6}  predicted")
    for k, v in sorted(done.items(), key=lambda kv: kv[1]["ks"]):
        raw = f"{v['ks_raw_ip']:.4f}" if "ks_raw_ip" in v else ""
        print(f"{k:<38} {v['split']:<5} {v['model']:<26} {v['ks']:>7.4f} {v['ks_ci95']:>7.4f} {v['ks_self']:>7.4f} {raw:>7} "
              f"{v['top1_var'] * 100:>5.1f}  {v['predicted']}")
    if not args.local and done:
        counts = {p: sum(v["predicted"].split(" ")[0] == p for v in done.values()) for p in ("Ada-ef", "band", "ours")}
        n_unc = sum("uncertain" in v["predicted"] for v in done.values())
        print(f"\n  predicted better score: Ada-ef {counts['Ada-ef']}, in band {counts['band']}, ours {counts['ours']} "
              f"(of {len(done)}; {n_unc} uncertain; band {BAND_LO}-{BAND_HI}, provisional)")
    print(f"\n  wrote {out_dir}/ks_survey.json")


if __name__ == "__main__":
    main()
