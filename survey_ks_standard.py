#!/usr/bin/env python3
"""
KS survey of the two standard ANN benchmark suites, to find standard (non-VIBE) datasets on
each side of the crossover band (updateAsOf260926.md §6: Ada-ef's Gaussian score ranks queries
better up to KS 0.044, our empirical score from 0.066).

Same KS as survey_ks_vibe.py (imported from it): 200 test queries, s = q.v over 20,000 corpus
points, KS statistic against Ada-ef's CLT-predicted Normal(q.mean, q'Cov q), unit-normalized
vectors, 95% interval, "(uncertain)" when the interval crosses a band edge. For inner-product
datasets the raw inner-product KS is reported too.

ann-benchmarks (ann-benchmarks.com; the suite most ANN papers use). Dense sets only:
  fashion-mnist-784, mnist-784, gist-960, glove-25/50/200, lastfm-64 (dot), nytimes-256
  (KS only: its duplicates make recall ill-posed, updateAsOf180926.md), coco-i2i/t2i-512.
  Already measured in the unified runs: sift-128, glove-100, deep-image-96. Skipped: kosarak and
  movielens10m (sparse sets, Jaccard similarity; the KS test does not apply).

Big-ANN (NeurIPS'21 billion-scale track, github.com/harsha-simhadri/big-ann-benchmarks):
  BIGANN (SIFT-1B, uint8), Deep1B, Text2Image-1B (inner product, OOD text-to-image), MS Turing,
  MS SpaceV (int8), FB-SSNPP (uint8). The files are `n, d` headers followed by rows; the
  benchmark's own 10M/100M subsets are prefixes of the 1B file, so this downloads only the first
  --bigann-rows rows (default 2M) with an HTTP range request, plus the public query file.
  Integer vectors are converted to float32 before normalizing.

Usage (on the server):
  python3 survey_ks_standard.py                        # both suites (~1 GB ann-benchmarks + ~5 GB Big-ANN)
  python3 survey_ks_standard.py --suite ann            # ann-benchmarks only
  python3 survey_ks_standard.py --datasets gist-960-euclidean,deep1b
"""

import os, json, argparse
from datetime import datetime

import numpy as np
import requests
from tqdm import tqdm

import survey_ks_vibe as sv

AB = "http://ann-benchmarks.com"
COCO = "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3"
# name -> (url, split, what, inner-product?)
ANN_BENCHMARKS = {
    "fashion-mnist-784-euclidean": (f"{AB}/fashion-mnist-784-euclidean.hdf5", "ID", "image pixels", False),
    "mnist-784-euclidean":         (f"{AB}/mnist-784-euclidean.hdf5", "ID", "image pixels", False),
    "gist-960-euclidean":          (f"{AB}/gist-960-euclidean.hdf5", "ID", "GIST image descriptor", False),
    "glove-25-angular":            (f"{AB}/glove-25-angular.hdf5", "ID", "word, GloVe", False),
    "glove-50-angular":            (f"{AB}/glove-50-angular.hdf5", "ID", "word, GloVe", False),
    "glove-200-angular":           (f"{AB}/glove-200-angular.hdf5", "ID", "word, GloVe", False),
    "lastfm-64-dot":               (f"{AB}/lastfm-64-dot.hdf5", "ID", "music recsys, matrix factorization", True),
    "nytimes-256-angular":         (f"{AB}/nytimes-256-angular.hdf5", "ID", "text, bag-of-words (KS only)", False),
    "coco-i2i-512-angular":        (f"{COCO}/coco-i2i-512-angular.hdf5", "ID", "image, CLIP", False),
    "coco-t2i-512-angular":        (f"{COCO}/coco-t2i-512-angular.hdf5", "OOD", "text-to-image, CLIP", False),
}
FB = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks"
YX = "https://storage.yandexcloud.net/yandex-research/ann-datasets"
MS = "https://comp21storage.z5.web.core.windows.net/comp21"
# name -> (base url, base file, query file, dtype, dim, split, what, inner-product?)
BIG_ANN = {
    "bigann":     (f"{FB}/bigann", "base.1B.u8bin", "query.public.10K.u8bin", "uint8", 128, "ID", "SIFT descriptor (SIFT-1B)", False),
    "deep1b":     (f"{YX}/DEEP", "base.1B.fbin", "query.public.10K.fbin", "float32", 96, "ID", "CNN image features", False),
    "text2image": (f"{YX}/T2I", "base.1B.fbin", "query.public.100K.fbin", "float32", 200, "OOD", "text-to-image, SE-ResNeXt", True),
    "msturing":   (f"{MS}/MSFT-TURING-ANNS", "base1b.fbin", "query100K.fbin", "float32", 100, "ID", "web text, Turing", False),
    "msspacev":   (f"{MS}/spacev1b", "spacev1b_base.i8bin", "query.i8bin", "int8", 100, "ID", "web text, SpaceV (int8)", False),
    "ssnpp":      (FB, "FB_ssnpp_database.u8bin", "FB_ssnpp_public_queries.u8bin", "uint8", 256, "ID", "image, SSNPP (uint8)", False),
}


def download(url, dest, max_bytes=None):
    """Plain or prefix (HTTP range) download with resume; skips files already complete."""
    if os.path.exists(dest):
        return dest
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    if max_bytes is not None and done >= max_bytes:
        os.replace(part, dest)
        return dest
    end = "" if max_bytes is None else str(max_bytes - 1)
    headers = {"Range": f"bytes={done}-{end}"} if (done or max_bytes) else {}
    with requests.get(url, stream=True, timeout=120, headers=headers, allow_redirects=True) as r:
        if (done or max_bytes) and r.status_code != 206:
            raise RuntimeError(f"{url}: server ignored the range request (HTTP {r.status_code})")
        r.raise_for_status()
        total = (max_bytes if max_bytes else int(r.headers.get("Content-Length", 0)) + done)
        with open(part, "ab" if done else "wb") as f, \
                tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(part, dest)
    return dest


def bin_loader(base_path, query_path, dtype, dim, rows):
    """Big-ANN .xbin files: int32 n, int32 d, then n*d values. The base prefix has `rows` rows."""
    def load():
        n_q, d_q = np.fromfile(query_path, dtype=np.int32, count=2)
        assert d_q == dim, f"query dim {d_q} != {dim}"
        q = np.memmap(query_path, dtype=dtype, mode="r", offset=8, shape=(int(n_q), dim))
        base = np.memmap(base_path, dtype=dtype, mode="r", offset=8, shape=(rows, dim))
        return base, np.asarray(q), {"source": "big-ann prefix", "rows": rows}, (lambda: None)
    return load


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["ann", "bigann", "both"], default="both")
    ap.add_argument("--datasets", default="", help="comma list of names (default: whole suite)")
    ap.add_argument("--bigann-rows", type=int, default=2_000_000, help="rows of each Big-ANN base prefix")
    ap.add_argument("--data-dir", default="standard_data")
    ap.add_argument("--n-queries", type=int, default=sv.N_QUERIES)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    sv.N_QUERIES = args.n_queries
    os.makedirs(args.data_dir, exist_ok=True)

    wanted = [s.strip() for s in args.datasets.split(",") if s.strip()]
    jobs = []
    if args.suite in ("ann", "both"):
        jobs += [("ann-benchmarks", n) for n in ANN_BENCHMARKS if not wanted or n in wanted]
    if args.suite in ("bigann", "both"):
        jobs += [("big-ann", n) for n in BIG_ANN if not wanted or n in wanted]

    out_dir = f"results_ks_survey_standard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for suite, name in jobs:
        try:
            if suite == "ann-benchmarks":
                url, split, what, ip = ANN_BENCHMARKS[name]
                path = download(url, os.path.join(args.data_dir, name + ".hdf5"))
                loader, allow_int = sv.h5_loader(path), False
            else:
                base_url, bfn, qfn, dtype, dim, split, what, ip = BIG_ANN[name]
                item = np.dtype(dtype).itemsize
                bpath = download(f"{base_url}/{bfn}", os.path.join(args.data_dir, f"{name}_base_{args.bigann_rows}.bin"),
                                 max_bytes=8 + args.bigann_rows * dim * item)
                qpath = download(f"{base_url}/{qfn}", os.path.join(args.data_dir, f"{name}_query.bin"))
                loader, allow_int = bin_loader(bpath, qpath, dtype, dim, args.bigann_rows), True
            if args.download_only:
                continue
            print(f"  {suite}: {name} ...", flush=True)
            r = sv.survey_source(loader, ip, np.random.default_rng(sv.SEED), allow_int=allow_int)
        except Exception as e:                 # one bad source must not stop the survey
            r, split, what = dict(error=repr(e)), locals().get("split", "?"), locals().get("what", "?")
        r.update(suite=suite, split=split, model=what)
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

    done = {k: v for k, v in results.items() if "ks" in v}
    print(f"\n{'dataset':<30} {'suite':<15} {'split':<5} {'what':<34} {'KS':>7} {'±95%':>7} {'self':>7} {'rawIP':>7}  predicted")
    for k, v in sorted(done.items(), key=lambda kv: kv[1]["ks"]):
        raw = f"{v['ks_raw_ip']:.4f}" if "ks_raw_ip" in v else ""
        print(f"{k:<30} {v['suite']:<15} {v['split']:<5} {v['model']:<34} {v['ks']:>7.4f} {v['ks_ci95']:>7.4f} "
              f"{v['ks_self']:>7.4f} {raw:>7}  {v['predicted']}")
    counts = {p: sum(v["predicted"].split(" ")[0] == p for v in done.values()) for p in ("Ada-ef", "band", "ours")}
    print(f"\n  predicted better score: Ada-ef {counts['Ada-ef']}, in band {counts['band']}, ours {counts['ours']} "
          f"(of {len(done)}; band {sv.BAND_LO}-{sv.BAND_HI})")
    failed = [k for k, v in results.items() if "ks" not in v]
    if failed:
        print(f"  not measured: {', '.join(failed)} (see {out_dir}/ks_survey.json)")
    print(f"\n  wrote {out_dir}/ks_survey.json")


if __name__ == "__main__":
    main()
