#!/usr/bin/env python3
"""
Unified benchmark: ours vs Ada-ef vs fixed ef, one frozen protocol for every dataset
(PAPER_PLAN.md). Replaces the per-dataset benchmark_*.py scripts for the paper's numbers.

Protocol (matches the Ada-ef paper, SIGMOD 2026, section 7.1, unless noted):
  - HNSW (this repo's vendored HNSWlib): M=16, ef_construction=500, unit-normalized vectors,
    L2 space (ranking identical to cosine).
  - K=1000 for MS MARCO-family and LAION, K=100 otherwise. Target recall 0.95. ef cap 5000.
  - Cost = distance computations per query: HNSW traversal + our centroid probe (K_clusters).
  - Two calibration settings, both methods always calibrated on the same set:
      P (paper): 200 vectors sampled from the corpus (left in the index), as the paper does.
      R (real) : held-out real queries, disjoint from the test set.
    The test set is the same for P and R within a dataset.
  - Ada-ef runs through its own code (AdaEfPaperScorer / AdaEfPaperSketch /
    adaptive_search_knn_paper), in two variants: as shipped (no WAE floor) and with
    Algorithm 1's max(ef, WAE) floor applied to its ef-estimation table.
  - Ours: K_clusters in {1, 8, 50} x {Isotonic, Mean, P90, P70}. Pre-registered headline
    configuration: K_clusters=1, Isotonic. The rest is the secondary frontier.
  - Fixed ef: HNSW at a grid of fixed ef values (a hindsight reference, not a baseline the
    paper claims to beat).
  - Reported per method: DC, mean recall, p1/p5 recall, target-hit rate, avg ef, plus per-query
    arrays (recall, ef, DC) for later analysis.

Memory: the corpus is never loaded whole. Everything before the index (statistics, KS, ground
truth, cluster bins) streams chunks from disk, and the index is built chunk by chunk, so peak
RAM is about the index itself (Cohere 9.5M x 1024: ~40 GB; LAION 20 shards: ~44 GB). Ada-ef's
dataset statistics are computed in the same streaming pass and handed to its unmodified
estimator via AdaEfPaperScorer.from_stats_file (needs the rebuilt extension). On datasets small
enough to hold in RAM, the script also builds the estimator the original way and checks both
give the same query scores.

Usage:
  python3 benchmark_unified.py --dataset sift128 --smoke     # ~minutes: end-to-end check
  python3 benchmark_unified.py --dataset sift128              # both settings, P then R
  python3 benchmark_unified.py --dataset cohere1024 --settings P
Datasets: glove100 deepimage96 sift128 dbpedia1536 yambda msmarco384 cohere1024 laion_i2i
          vibe_landmark_dino vibe_inaturalist_resnet vibe_yahoo_minilm vibe_imagenet_align
          gist960 fashionmnist784
          lastfm64 coco_i2i coco_t2i                               (validation round 2, ann-benchmarks)
          deep1b bigann msturing text2image msspacev ssnpp         (validation round 2, Big-ANN 2M prefixes)

Since validation round 2 (PAPER_PLAN.md) it also records, per method and setting:
  - wall-clock latency per query (single-threaded search call, microseconds), and the cost saving
    against a tuned fixed ef at the same mean recall in both latency and DC;
  - offline cost: calibration time and the memory of what each method keeps;
  - calibration diagnostics: share of queries at the ef floor / capped, headroom, distinct scores;
  - the calibration-time choice between the two scores (keep ours if |rho_ours(K=1)| >= |rho_Ada|),
    as rows "Choice (with Ada-ef as shipped)" and "Choice (with Ada-ef WAE floor)".
Latency comes from a separate timing pass (--lat-rounds rounds over the fixed-ef grid and the headline
methods, method order rotated every round, per-query median), so a slowdown of the machine at one
moment does not land on one method. It is still only comparable within one run: run one benchmark
at a time on an otherwise idle machine.
"""

import os, re, sys, json, time, gzip, glob, pickle, struct, argparse, subprocess
from datetime import datetime

import numpy as np

DATASETS = ["glove100", "deepimage96", "sift128", "dbpedia1536", "yambda", "msmarco384",
            "cohere1024", "laion_i2i",
            "vibe_landmark_dino", "vibe_inaturalist_resnet", "vibe_yahoo_minilm", "vibe_imagenet_align",
            "gist960", "fashionmnist784",
            "lastfm64", "coco_i2i", "coco_t2i",
            "deep1b", "bigann", "msturing", "text2image", "msspacev", "ssnpp"]

# Validation round 2 (PAPER_PLAN.md), downloaded by survey_ks_standard.py into standard_data/.
# ann-benchmarks files: name -> (file, note). KS from the 200-query survey (updateAsOf280926.md §4).
ROUND2_ANN = {
    "lastfm64": ("lastfm-64-dot.hdf5", "KS 0.216, predicted ours; inner-product task run on normalised "
                                       "vectors (cosine), a stated deviation"),
    "coco_i2i": ("coco-i2i-512-angular.hdf5", "KS 0.046, band"),
    "coco_t2i": ("coco-t2i-512-angular.hdf5", "KS 0.056, band (OOD: text queries, image corpus)"),
}
# Big-ANN NeurIPS'21 first-2M-row prefixes: name -> (dtype, dim, note)
ROUND2_BIGANN = {
    "deep1b":     ("float32", 96, "KS 0.067, predicted ours (uncertain)"),
    "bigann":     ("uint8", 128, "KS 0.029, predicted Ada-ef"),
    "msturing":   ("float32", 100, "KS 0.010, predicted Ada-ef"),
    "text2image": ("float32", 200, "KS 0.032, predicted Ada-ef; OOD, inner-product task run normalised"),
    "msspacev":   ("int8", 100, "KS 0.024, predicted Ada-ef"),
    "ssnpp":      ("uint8", 256, "KS 0.007, predicted Ada-ef"),
}
BIGANN_ROWS = 2_000_000
N_TEST_MAX = 10000                # large query sets (Last.fm 50K, Big-ANN 10K-100K) test on 10,000

# VIBE datasets (huggingface.co/datasets/vector-index-bench/vibe, fetched by survey_ks_vibe.py into
# vibe_data/), chosen from the KS survey to test the crossover rule out of sample
# (updateAsOf260926.md section 7): two predicted to favour our score (KS >= 0.066) and two inside
# the band. name -> (VIBE file, has a `learn` set of real OOD queries)
VIBE_SETS = {
    "vibe_landmark_dino":      ("landmark-dino-768-cosine", False),         # KS 0.076, predicted ours
    "vibe_inaturalist_resnet": ("inaturalist-resnet-2048-cosine", False),   # KS 0.115, predicted ours
    "vibe_yahoo_minilm":       ("yahoo-minilm-384-normalized", False),      # KS 0.054, band
    "vibe_imagenet_align":     ("imagenet-align-640-normalized", True),     # KS 0.058, band (OOD, text-to-image)
}

ap = argparse.ArgumentParser(description="Unified ours / Ada-ef / fixed-ef benchmark")
ap.add_argument("--dataset", required=True, choices=DATASETS)
ap.add_argument("--settings", default="P,R", help="comma list of P (paper calibration) and R (real queries)")
ap.add_argument("--cohere-files", type=int, default=5, help="Cohere passage files 00..N-1 to use (paper: 10)")
ap.add_argument("--laion-shards", type=int, default=20, help="LAION image shards 0..N-1 to use (paper: 31)")
ap.add_argument("--smoke", action="store_true", help="tiny query sets and K=1 only, to check the pipeline")
ap.add_argument("--lat-rounds", type=int, default=3,
                help="rounds of the latency pass (fixed ef + headline methods; per-query median)")
ap.add_argument("--ablation", action="store_true",
                help="probe-length ablation: Ada-ef with a 100-distance probe and ours with 1025 (own results folder)")
ap.add_argument("--quick", action="store_true", help="K_clusters=1 and the Isotonic recipe only")
ap.add_argument("--group-table", action="store_true",
                help="also run PercEF's score with Ada-ef's group-average ef table (own results folder)")
ap.add_argument("--export-darth", metavar="DIR", default=None,
                help="write this dataset in DARTH's file layout under DIR and exit (needs an earlier run's caches)")
ap.add_argument("--target-recall", type=float, default=None,
                help="sweep: target recall other than the frozen 0.95 (own results folder and caches)")
ap.add_argument("--k", type=int, default=None,
                help="sweep: number of neighbours other than the dataset's default (own results folder and caches)")
ap.add_argument("--fixes", action="store_true",
                help="also run the two candidate fixes (updateAsOf071026.md §6): the ef table scaled to meet the "
                     "target on the calibration queries, and a shorter probe (--short-probe); own results folder")
ap.add_argument("--short-probe", type=int, default=30, help="probe length of the shorter-probe variant (--fixes)")
ap.add_argument("--seed", type=int, default=None,
                help="seed repeat: another random draw of the R/test split, the P calibration points and the KS "
                     "sample (default 42); same index; own results folder and caches")
args = ap.parse_args()
SWEEP = args.target_recall is not None or args.k is not None or args.fixes or args.seed is not None
SETTINGS = [s.strip().upper() for s in args.settings.split(",") if s.strip()]
assert all(s in ("P", "R") for s in SETTINGS), "--settings takes P and/or R"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SEED_TAG = f"_s{args.seed}" if args.seed is not None else ""
CACHE_TAG = ((f"_t{args.target_recall:g}" if args.target_recall is not None else "")
             + (f"_k{args.k}" if args.k is not None else "") + SEED_TAG)
SWEEP_TAG = CACHE_TAG + ("_fixes" if args.fixes else "")
RESULTS_DIR = (f"results_unified_{args.dataset}{'_smoke' if args.smoke else ''}"
               f"{'_ablation' if (args.ablation or args.group_table) else ''}"
               f"{'_sweep' + SWEEP_TAG if SWEEP else ''}_{TIMESTAMP}")
if not args.export_darth:
    os.makedirs(RESULTS_DIR, exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')


class Logger:
    def __init__(self, path):
        self.terminal, self.log = sys.stdout, open(path, "a", encoding="utf-8")
    def write(self, m):
        self.terminal.write(m); self.terminal.flush(); self.log.write(m); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


if not args.export_darth:
    sys.stdout = Logger(os.path.join(RESULTS_DIR, "benchmark_unified.log"))

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kstest
from sklearn.cluster import MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chao_hybrid_ada_ef'))
import chao_hybrid_ada_ef_cpp as hnsw

# ═══════════════════════════════════════════════════════════════════════
#  Frozen protocol
# ═══════════════════════════════════════════════════════════════════════
SEED = 42 if args.seed is None else args.seed
M, EF_CONSTRUCTION = 16, 500
TARGET_RECALL = 0.95 if args.target_recall is None else args.target_recall
EF_CAP = 5000
PROBE_COUNT, NUM_BINS, QUANTILE_STEP, STATICS_LENGTH = 100, 5, 1e-3, 1025
BIN_WEIGHTS = [float(100.0 * np.exp(-i)) for i in range(NUM_BINS)]
K_SWEEP = [1] if (args.smoke or args.quick) else [1, 8, 50]
RECIPES = ["Isotonic"] if args.quick else ["Isotonic", "Mean", "P90", "P70"]
DEFAULT_CONFIG = "Ours (K=1, Isotonic)"
N_CALIB_P = 50 if args.smoke else 200
N_CALIB_R_FULL = 2000             # dataset splits never depend on --smoke
N_CALIB_R = 300 if args.smoke else N_CALIB_R_FULL
N_TEST_SMOKE = 300
CHUNK = 100_000
KS_QUERIES, KS_SAMPLE, KS_POOL = 30, 20000, 200_000
VERIFY_MAX_BYTES = 7e9          # build Ada-ef's estimator the original way too, if the corpus fits


def ef_grid(k):
    if k < 100:                     # sweep only (k = 10): 5-step up to 100, then the usual 50-step
        return list(range(k, 100, 5)) + list(range(100, EF_CAP + 1, 50))
    return list(range(k, EF_CAP + 1, 50 if k <= 100 else 100))


def fixed_efs(k):
    # Dense where the methods land (updateAsOf290926.md: the fixed-ef reference is interpolated
    # between grid points, so closer points leave less to interpolation).
    if k < 100:                     # sweep only
        return [k, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 125, 150, 200, 250, 300, 400, 500, 800, 1000, 2000]
    if k <= 100:
        return [100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000, 1250, 1500, 2000, 3000, 5000]
    return [1000, 1125, 1250, 1375, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000]


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


# ═══════════════════════════════════════════════════════════════════════
#  Streaming corpus: one or more on-disk arrays (np.memmap or h5py datasets),
#  optional held-out rows. Labels are global row ids everywhere.
# ═══════════════════════════════════════════════════════════════════════
class Corpus:
    def __init__(self, parts, keep=None):
        self.parts = parts
        self.offsets = np.cumsum([0] + [p.shape[0] for p in parts])
        self.n_total = int(self.offsets[-1])
        self.dim = int(parts[0].shape[1])
        self.keep = keep
        self.n = self.n_total if keep is None else int(keep.sum())

    def kept_ids(self):
        return np.arange(self.n_total, dtype=np.int64) if self.keep is None else np.flatnonzero(self.keep)

    def chunks(self, size=CHUNK):
        for pi, part in enumerate(self.parts):
            base = int(self.offsets[pi])
            for a in range(0, part.shape[0], size):
                b = min(a + size, part.shape[0])
                ids = np.arange(base + a, base + b, dtype=np.int64)
                x = normalize(part[a:b])
                if self.keep is not None:
                    m = self.keep[base + a:base + b]
                    ids, x = ids[m], x[m]
                if len(ids):
                    yield ids, x

    def rows(self, ids):
        """Normalized rows for arbitrary global ids (small sets), in the order given."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), self.dim), dtype=np.float32)
        for pi, part in enumerate(self.parts):
            lo, hi = self.offsets[pi], self.offsets[pi + 1]
            sel = np.flatnonzero((ids >= lo) & (ids < hi))
            if len(sel):
                order = np.argsort(ids[sel])
                local = (ids[sel][order] - lo)
                out[sel[order]] = normalize(part[local.tolist()] if hasattr(part, "id") else part[local])
        return out


def split_queries(q, n_calib, rng):
    """Held-out real queries: first n_calib of a seeded permutation calibrate, the rest test."""
    perm = rng.permutation(len(q))
    return q[perm[n_calib:]], q[perm[:n_calib]]


def held_out_split(n_total, n_calib, n_test, rng):
    """Remove n_calib + n_test random rows from the corpus (no trivial self-matches)."""
    held = rng.choice(n_total, n_calib + n_test, replace=False)
    keep = np.ones(n_total, dtype=bool)
    keep[held] = False
    return keep, held[n_calib:], held[:n_calib]


def load_dataset(name):
    """Returns dict: corpus, k, test_q, calib_r, index_path, reuse, tag, notes."""
    import h5py
    if args.seed is not None and name in ("laion_i2i", "yambda"):
        # their queries are held out of the corpus (held_out_split): another seed changes the corpus,
        # and the cached index would contain the new test queries
        sys.exit(f"--seed is not supported for {name}: its queries are taken from the corpus")
    rng = np.random.default_rng(SEED)
    n_r = N_CALIB_R_FULL
    if name in ("glove100", "deepimage96", "sift128", "dbpedia1536", "gist960", "fashionmnist784"):
        # gist960 / fashionmnist784: standard ann-benchmarks sets the KS survey put on our side of the
        # band (0.091, 0.073; updateAsOf280926.md), fetched by survey_ks_standard.py into standard_data/.
        files = {"glove100": "glove-100-angular.hdf5", "deepimage96": "deep-image-96-angular.hdf5",
                 "sift128": "sift-128-euclidean.hdf5", "dbpedia1536": "dbpedia-openai-1000k-angular.hdf5",
                 "gist960": os.path.join("standard_data", "gist-960-euclidean.hdf5"),
                 "fashionmnist784": os.path.join("standard_data", "fashion-mnist-784-euclidean.hdf5")}
        reuse = {"deepimage96": "custom_full_deep_image_efc500.index", "sift128": "sift128_efc500_m16.index",
                 "dbpedia1536": "dbpedia_openai1536_efc500_m16.index"}
        if not os.path.exists(files[name]):
            sys.exit(f"{files[name]} not found (gist960/fashionmnist784: run survey_ks_standard.py --suite ann --download-only)")
        f = h5py.File(files[name], "r")
        q = normalize(f["test"][:])
        n_rc = min(n_r, len(q) * 3 // 10)    # GIST ships only 1,000 test queries; the 10,000-query sets keep 2,000
        test, calib = split_queries(q, n_rc, rng)
        return dict(corpus=Corpus([f["train"]]), k=100, test_q=test, calib_r=calib, tag=name,
                    index_path=reuse.get(name, os.path.join("unified_cache", name, "index_m16_efc500.index")),
                    reuse=name in reuse,
                    notes=f"ann-benchmarks file; {len(q)} test queries, {n_rc} held out for R-calibration")
    if name in VIBE_SETS:
        vname, has_learn = VIBE_SETS[name]
        path = os.path.join("vibe_data", vname + ".hdf5")
        if not os.path.exists(path):
            sys.exit(f"{path} not found; run survey_ks_vibe.py --download-only --datasets {vname}")
        f = h5py.File(path, "r")
        q = normalize(f["test"][:])
        if has_learn:
            # OOD: all 1,000 test queries are tested; R-calibration draws from `learn`, a larger
            # sample of the same (text) query distribution.
            learn = f["learn"]
            pick = np.sort(rng.choice(learn.shape[0], n_r, replace=False))
            test, calib, how = q, normalize(learn[pick.tolist()]), f"all {len(q)} test queries; R-calibration: {n_r} `learn` queries"
        else:
            # In-distribution: VIBE ships only 1,000 test queries, so 30% calibrate R (as for Cohere).
            n_rc = len(q) * 3 // 10
            test, calib = split_queries(q, n_rc, rng)
            how = f"{len(q)} test queries, {n_rc} held out for R-calibration"
        return dict(corpus=Corpus([f["train"]]), k=100, test_q=test, calib_r=calib, tag=name,
                    index_path=os.path.join("unified_cache", name, "index_m16_efc500.index"), reuse=False,
                    notes=f"VIBE {vname}; {how}")
    if name == "msmarco384":
        f = h5py.File("msmarco-8.8M-minilm-384d.hdf5", "r")
        test = normalize(np.load("msmarco_qemb_validation.npz")["emb"])
        train = np.load("msmarco_qemb_train.npz")["emb"]
        calib = normalize(train[rng.choice(len(train), n_r, replace=False)])
        return dict(corpus=Corpus([f["embeddings"]]), k=1000, test_q=test, calib_r=calib, tag=name,
                    index_path=os.path.join("unified_cache", name, "index_m16_efc500.index"), reuse=False,
                    notes="MiniLM-384 stand-in for the paper's MS MARCO V1 (OpenAI-1536); test = dev queries, "
                          "R-calibration = train queries")
    if name == "cohere1024":
        d = "cohere_msmarco_v21_npy"
        files = sorted(glob.glob(os.path.join(d, "msmarco_v2.1_doc_segmented_*.npy")))[:args.cohere_files]
        if len(files) < args.cohere_files:
            sys.exit(f"Need {args.cohere_files} Cohere files in {d}/, found {len(files)}; run download_unified_data.py")
        parts = [np.load(p, mmap_mode="r") for p in files]
        with gzip.open(os.path.join(d, "queries.jsonl.gz"), "rt", encoding="utf-8") as fq:
            q = np.array([json.loads(line)["emb"] for line in fq], dtype=np.float32)
        n_rc = min(n_r, len(q) * 3 // 10)          # only 1,677 queries exist: 30% calibrate R
        test, calib = split_queries(normalize(q), n_rc, rng)
        tag = f"cohere1024_f{len(files)}"
        return dict(corpus=Corpus(parts), k=1000, test_q=test, calib_r=calib, tag=tag,
                    index_path=os.path.join("unified_cache", tag, "index_m16_efc500.index"), reuse=False,
                    notes=f"authors' source files 00..{len(files) - 1:02d} of 10 (subset: server RAM); "
                          f"{len(q)} queries, {n_rc} held out for R-calibration")
    if name == "laion_i2i":
        files = [os.path.join("laion_i2i_subset", "shards", f"img_emb_{i}.npy") for i in range(args.laion_shards)]
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            sys.exit(f"Missing LAION shards ({len(missing)}), e.g. {missing[0]}; run download_unified_data.py")
        parts = [np.load(p, mmap_mode="r") for p in files]
        corpus = Corpus(parts)
        n_test = 10000
        keep, test_ids, calib_ids = held_out_split(corpus.n_total, n_r, n_test, rng)
        test, calib = corpus.rows(test_ids), corpus.rows(calib_ids)
        tag = f"laion_i2i_s{args.laion_shards}"
        return dict(corpus=Corpus(parts, keep), k=1000, test_q=test, calib_r=calib, tag=tag,
                    index_path=os.path.join("unified_cache", tag, "index_m16_efc500.index"), reuse=False,
                    notes=f"authors' source, shards 0..{args.laion_shards - 1} of 31 (subset: server RAM); "
                          f"{n_test} test + {n_r} R-calibration rows held out of the corpus")
    if name == "yambda":
        part = np.load("yambda_audio_corpus.npy", mmap_mode="r")
        corpus = Corpus([part])
        n_test = 10000
        keep, test_ids, calib_ids = held_out_split(corpus.n_total, n_r, n_test, rng)
        test, calib = corpus.rows(test_ids), corpus.rows(calib_ids)
        return dict(corpus=Corpus([part], keep), k=100, test_q=test, calib_r=calib, tag="yambda",
                    index_path=os.path.join("unified_cache", "yambda", "index_m16_efc500.index"), reuse=False,
                    notes=f"no query file: {n_test} test + {n_r} R-calibration tracks held out of the corpus")
    if name in ROUND2_ANN:
        fn, note = ROUND2_ANN[name]
        path = os.path.join("standard_data", fn)
        if not os.path.exists(path):
            sys.exit(f"{path} not found; run survey_ks_standard.py --suite ann --download-only")
        f = h5py.File(path, "r")
        q = normalize(f["test"][:])
        n_rc = min(n_r, len(q) * 3 // 10)
        test, calib = split_queries(q, n_rc, rng)
        test = test[:N_TEST_MAX]
        return dict(corpus=Corpus([f["train"]]), k=100, test_q=test, calib_r=calib, tag=name,
                    index_path=os.path.join("unified_cache", name, "index_m16_efc500.index"), reuse=False,
                    notes=f"ann-benchmarks {fn}; {len(q)} test queries, {n_rc} R-calibration, {len(test)} tested; {note}")
    if name in ROUND2_BIGANN:
        dtype, dim, note = ROUND2_BIGANN[name]
        bpath = os.path.join("standard_data", f"{name}_base_{BIGANN_ROWS}.bin")
        qpath = os.path.join("standard_data", f"{name}_query.bin")
        if not (os.path.exists(bpath) and os.path.exists(qpath)):
            sys.exit(f"{bpath} / {qpath} not found; run survey_ks_standard.py --suite bigann --download-only")
        item = np.dtype(dtype).itemsize
        rows = (os.path.getsize(bpath) - 8) // (dim * item)     # the header says 1B; the file is a prefix
        assert rows == BIGANN_ROWS, f"{bpath} has {rows} rows, expected {BIGANN_ROWS}"
        n_q, d_q = np.fromfile(qpath, dtype=np.int32, count=2)
        assert d_q == dim, f"query dim {d_q} != {dim}"
        base = np.memmap(bpath, dtype=dtype, mode="r", offset=8, shape=(rows, dim))
        q = normalize(np.memmap(qpath, dtype=dtype, mode="r", offset=8, shape=(int(n_q), dim)))
        test, calib = split_queries(q, n_r, rng)
        test = test[:N_TEST_MAX]
        return dict(corpus=Corpus([base]), k=100, test_q=test, calib_r=calib, tag=f"{name}_{rows // 1_000_000}M",
                    index_path=os.path.join("unified_cache", f"{name}_{rows // 1_000_000}M", "index_m16_efc500.index"),
                    reuse=False,
                    notes=f"Big-ANN {name}, first {rows} rows of the 1B file (subset); {int(n_q)} public queries, "
                          f"{n_r} R-calibration, {len(test)} tested; {note}")
    raise ValueError(name)


# ═══════════════════════════════════════════════════════════════════════
#  Streaming passes
# ═══════════════════════════════════════════════════════════════════════
def stats_pass(corpus, pool_ids, p_ids):
    """One pass: mean, unbiased covariance (float64), a uniform KS pool, P-calibration rows."""
    d = corpus.dim
    s, xtx, n = np.zeros(d), np.zeros((d, d)), 0
    pool_sorted, p_sorted = np.sort(pool_ids), np.sort(p_ids)
    pool, p_rows = {}, {}
    for ids, x in corpus.chunks():
        s += x.sum(axis=0, dtype=np.float64)
        xtx += (x.T @ x).astype(np.float64)
        n += len(ids)
        for sorted_ids, store in ((pool_sorted, pool), (p_sorted, p_rows)):
            hit = np.isin(ids, sorted_ids, assume_unique=True)
            for i, row in zip(ids[hit], x[hit]):
                store[int(i)] = row
    mean = s / n
    cov = (xtx - n * np.outer(mean, mean)) / (n - 1)
    return mean, cov, n, np.stack([pool[int(i)] for i in pool_ids]), np.stack([p_rows[int(i)] for i in p_ids])


def write_ada_stats(path, mean, cov):
    """Ada-ef's own Estimator::serialize layout, read back by hnswdis::load_estimator_from_file."""
    name = b"CosineDistanceEstimator"
    c = np.asarray(cov, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(name))); f.write(name)
        f.write(struct.pack("<ii", c.shape[0], c.shape[1])); f.write(c.tobytes(order="F"))
        f.write(struct.pack("<i", len(mean))); f.write(np.asarray(mean, dtype="<f4").tobytes())
        f.write(struct.pack("<i", len(mean))); f.write(np.diag(c).astype("<f4").tobytes())


def ground_truth_pass(corpus, queries, k, qbatch=1024):
    """Exact top-k by inner product on normalized vectors, streamed over the corpus."""
    m = len(queries)
    best_s = np.full((m, k), -np.inf, dtype=np.float32)
    best_i = np.full((m, k), -1, dtype=np.int64)
    for ids, x in corpus.chunks():
        for a in range(0, m, qbatch):
            b = min(a + qbatch, m)
            sims = queries[a:b] @ x.T
            if sims.shape[1] > k:
                top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
                s_top, i_top = np.take_along_axis(sims, top, axis=1), ids[top]
            else:
                s_top, i_top = sims, np.broadcast_to(ids, sims.shape)
            cs = np.concatenate([best_s[a:b], s_top], axis=1)
            ci = np.concatenate([best_i[a:b], i_top], axis=1)
            keep = np.argpartition(-cs, k - 1, axis=1)[:, :k]
            best_s[a:b] = np.take_along_axis(cs, keep, axis=1)
            best_i[a:b] = np.take_along_axis(ci, keep, axis=1)
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_i, order, axis=1)


def cluster_pass(corpus, centroids_by_k):
    """Squared distance of every corpus point to its nearest centroid, for every K at once.
    Also returns the seconds spent per K (the shared corpus read is not included)."""
    dists = {k: [] for k in centroids_by_k}
    labels = {k: [] for k in centroids_by_k}
    secs = {k: 0.0 for k in centroids_by_k}
    for _, x in corpus.chunks():
        for k, c in centroids_by_k.items():
            t0 = time.time()
            d2 = cdist(x, c, metric="sqeuclidean")
            lab = np.argmin(d2, axis=1)
            labels[k].append(lab.astype(np.int32))
            dists[k].append(d2[np.arange(len(x)), lab].astype(np.float32))
            secs[k] += time.time() - t0
    pcts = [QUANTILE_STEP * (i + 1) * 100 for i in range(NUM_BINS)]
    bins = {}
    for k in centroids_by_k:
        t0 = time.time()
        dk, lk = np.concatenate(dists[k]), np.concatenate(labels[k])
        b = np.zeros((k, NUM_BINS), dtype=np.float32)
        for c in range(k):
            sel = dk[lk == c]
            b[c] = np.percentile(sel, pcts) if len(sel) else np.array([0.05, 0.1, 0.15, 0.2, 0.25])
        bins[k] = b
        secs[k] += time.time() - t0
    return bins, secs


def ks_fit(pool, mean, cov, queries, rng):
    vals = []
    for qi in rng.choice(len(queries), min(KS_QUERIES, len(queries)), replace=False):
        q = queries[qi].astype(np.float64)
        mu, var = float(q @ mean), float(q @ cov @ q)
        s = pool[rng.choice(len(pool), min(KS_SAMPLE, len(pool)), replace=False)].astype(np.float64) @ q
        vals.append(kstest(s, "norm", args=(mu, np.sqrt(max(var, 1e-12)))).statistic)
    return float(np.mean(vals)), [float(v) for v in vals]


# ═══════════════════════════════════════════════════════════════════════
#  Calibration helpers (unchanged logic from the earlier benchmark scripts)
# ═══════════════════════════════════════════════════════════════════════
def finite_or_none(x):
    """rho is undefined (NaN) when every calibration score is identical; JSON gets null, not NaN."""
    x = float(x)
    return x if np.isfinite(x) else None


def recall_of(labs, gt_row, k):
    return len(set(labs.tolist()) & set(gt_row[:k].tolist())) / k


def min_ef_sweep(idx, queries, gts, k, grid):
    out, capped = np.zeros(len(queries), dtype=np.float32), np.zeros(len(queries), dtype=bool)
    for i in range(len(queries)):
        for ef in grid:
            labs, _ = idx.search_knn_adaptive(queries[i], k, idx.entry_point, idx.max_level, ef)
            if recall_of(labs, gts[i], k) >= TARGET_RECALL:
                out[i] = ef
                break
        else:
            out[i], capped[i] = grid[-1], True
        if (i + 1) % 250 == 0:
            print(f"    ... {i + 1}/{len(queries)}", flush=True)
    return out, capped


def ada_target_recall_table(idx, scores_int, queries, gts, k, grid):
    """Ada-ef's offline table: per score group, smallest ef whose group-average recall hits the target."""
    table, wsum, tot = {}, 0, 0
    for s in np.unique(scores_int):
        sel = np.flatnonzero(scores_int == s)
        ef_found = grid[-1]
        for ef in grid:
            recs = [recall_of(idx.search_knn_adaptive(queries[i], k, idx.entry_point, idx.max_level, ef)[0], gts[i], k)
                    for i in sel]
            if np.mean(recs) >= TARGET_RECALL:
                ef_found = ef
                break
        table[int(s)] = int(ef_found)
        wsum += len(sel) * ef_found
        tot += len(sel)
    return table, int(wsum / tot)


def build_isotonic(scores_int, req, k):
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(scores_int, req)
    return [int(np.clip(v, k, EF_CAP)) for v in iso.predict(np.arange(int(scores_int.max()) + 1))]


def build_bucket(scores_int, req, how):
    agg = {"Mean": np.mean, "P90": lambda v: np.percentile(v, 90), "P70": lambda v: np.percentile(v, 70)}[how]
    return {int(s): int(agg(req[scores_int == s])) for s in np.unique(scores_int)}


def table_to_list(table, k):
    keys = sorted(table)
    out = []
    for s in range(keys[-1] + 1):
        if s in table:
            v = table[s]
        elif s <= keys[0]:
            v = table[keys[0]]
        else:
            lo = max(x for x in keys if x <= s); hi = min(x for x in keys if x >= s)
            v = table[lo] + (s - lo) / (hi - lo) * (table[hi] - table[lo])
        out.append(int(np.clip(v, k, EF_CAP)))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Online evaluation: per-query recall, ef and distance computations
# ═══════════════════════════════════════════════════════════════════════
def summarize(name, rec, efs, dcs, probe, lat, extra=None):
    r, lat = np.asarray(rec), np.asarray(lat, dtype=np.float64)
    row = dict(name=name, mean_r=float(r.mean()), p1=float(np.percentile(r, 1)), p5=float(np.percentile(r, 5)),
               pct_target=float(np.mean(r >= TARGET_RECALL) * 100), hnsw_dc=float(np.mean(dcs)),
               probe_dc=int(probe), total_dc=float(np.mean(dcs) + probe), avg_ef=float(np.mean(efs)),
               mean_lat_us=float(lat.mean()), p50_lat_us=float(np.percentile(lat, 50)),
               p99_lat_us=float(np.percentile(lat, 99)), qps=float(1e6 / lat.mean()),
               distinct_ef=int(len(np.unique(efs))))
    if extra:
        row.update(extra)
    return row, dict(recall=r.astype(np.float32), ef=np.asarray(efs, dtype=np.float32),
                     dc=(np.asarray(dcs, dtype=np.float64) + probe).astype(np.float32), lat_us=lat.astype(np.float32))


def run_queries(idx, test_q, test_gt, k, search):
    """search(i, q) -> (labels, ef used) for test query i. Latency is the wall-clock time of the
    search call alone (one thread; everything a method does per query must happen inside it)."""
    rec, efs, dcs, lat = [], [], [], []
    for i in range(len(test_q)):
        q = test_q[i]
        idx.reset_dist_count()
        t0 = time.perf_counter()
        labs, ef = search(i, q)
        lat.append((time.perf_counter() - t0) * 1e6)
        dcs.append(idx.get_dist_count())
        efs.append(ef)
        rec.append(recall_of(labs, test_gt[i], k))
    return rec, efs, dcs, lat


def timing_pass(searchers, test_q, rounds):
    """Latency only. Each round runs every method over all queries, one method after another, with
    the method order rotated every round so drift in machine speed hits all methods alike (blocks
    rather than per-query interleaving, so no method runs on a cache warmed by another's search of
    the same query). Returns each method's per-query median latency in microseconds."""
    names = list(searchers)
    lat = {n: np.zeros((rounds, len(test_q))) for n in names}
    for r in range(rounds):
        shift = (r * len(names)) // rounds
        for n in names[shift:] + names[:shift]:
            search = searchers[n]
            for i in range(len(test_q)):
                q = test_q[i]
                t0 = time.perf_counter()
                search(i, q)
                lat[n][r, i] = (time.perf_counter() - t0) * 1e6
        print(f"    latency round {r + 1}/{rounds} done", flush=True)
    return {n: np.median(v, axis=0) for n, v in lat.items()}


def set_latency(row, pq, lat):
    """Copies of row and per-query dict with latency fields taken from the timing pass."""
    lat = np.asarray(lat, dtype=np.float64)
    row = dict(row, mean_lat_us=float(lat.mean()), p50_lat_us=float(np.percentile(lat, 50)),
               p99_lat_us=float(np.percentile(lat, 99)), qps=float(1e6 / lat.mean()))
    return row, dict(pq, lat_us=lat.astype(np.float32))


def fixed_interp(fixed, goal_r, key):
    """A tuned fixed ef at mean recall goal_r, interpolated between the two fixed-ef rows around it
    in log(1 - recall) vs log(cost) space (p1/p5 linearly, at the same position). Linear
    interpolation in (recall, cost) overstates the fixed ef's cost, since cost grows ever faster as
    recall nears 1: +11% on average in a leave-one-out test on the fixed-ef grids, log-log -0.4%
    (analysis/rescore_scorecard.js --loo; updateAsOf290926.md). None outside the grid's recall range."""
    pts = sorted(fixed, key=lambda r: r["avg_ef"])
    x = lambda r: np.log(max(1.0 - r, 1e-6))
    for a, b in zip(pts, pts[1:]):
        if a["mean_r"] < goal_r <= b["mean_r"]:
            t = (x(goal_r) - x(a["mean_r"])) / (x(b["mean_r"]) - x(a["mean_r"]))
            if key in ("p1", "p5"):
                return a[key] + t * (b[key] - a[key])
            return float(np.exp(np.log(a[key]) + t * np.log(b[key] / a[key])))
    return None


def vs_fixed(row, fixed):
    """The scorecard entry: saving in DC and latency, and p1/p5 gain, against a fixed ef at row's recall."""
    out = {}
    for key, name in (("total_dc", "saving_dc_pct"), ("mean_lat_us", "saving_lat_pct")):
        f = fixed_interp(fixed, row["mean_r"], key)
        out[name] = None if f is None else (f - row[key]) / f * 100
    for key in ("p1", "p5"):
        f = fixed_interp(fixed, row["mean_r"], key)
        out[f"{key}_gain"] = None if f is None else row[key] - f
    # one ef for every test query = no adaptation: the method is itself a fixed ef (plus its probe)
    out["adapts"] = row.get("distinct_ef", 2) > 1
    return out


def load_offline_times(cache):
    path = os.path.join(cache, f"offline_times{CACHE_TAG}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_offline_time(cache, key, secs):
    """Offline steps are cached across runs; their first-run time is kept next to the cache."""
    t = load_offline_times(cache)
    t[key] = secs
    with open(os.path.join(cache, f"offline_times{CACHE_TAG}.json"), "w") as f:
        json.dump(t, f, indent=1)


def dc_at_quality(rows, metric, goal):
    pts = sorted(rows, key=lambda r: (r["total_dc"], -r[metric]))
    front, best = [], -np.inf
    for p in pts:
        if p[metric] > best:
            front.append(p); best = p[metric]
    if not front:
        return None
    if front[0][metric] >= goal:
        return dict(dc=front[0]["total_dc"], bound=True, via=[front[0]["name"]])
    for a, b in zip(front, front[1:]):
        if b[metric] >= goal:
            t = (goal - a[metric]) / (b[metric] - a[metric])
            return dict(dc=a["total_dc"] + t * (b["total_dc"] - a["total_dc"]), bound=False, via=[a["name"], b["name"]])
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def write_vecs(f, x, dtype):
    """fvecs / ivecs: each row is int32 dim followed by the values."""
    x = np.ascontiguousarray(x, dtype=dtype)
    rec = np.empty((len(x), x.shape[1] + 1), dtype=dtype)
    rec[:, 0] = np.frombuffer(np.int32(x.shape[1]).tobytes(), dtype=dtype)[0]
    rec[:, 1:] = x
    f.write(rec.tobytes())


def export_darth(spec, out_root):
    """Write the dataset in the layout darth/setup_darth.sh teaches DARTH's loader to read:
    <out_root>/<NAME>/base.fvecs, {train,validation,test}.fvecs, {split}.gt.ivecs (ids of the
    true top-K) and {split}.gtd.fvecs (their squared L2 distances). Train + validation = the
    R-calibration queries, test = the test queries, ground truth = the cached one the other
    methods were scored on. Vectors are unit-normalised, as in every other run."""
    corpus, K = spec["corpus"], spec["k"]
    gt_path = os.path.join("unified_cache", spec["tag"], f"gt_k{K}.npz")
    if not os.path.exists(gt_path):
        sys.exit(f"{gt_path} not found: run the benchmark on this dataset once first")
    g = np.load(gt_path)
    name = "CUSTOM_" + args.dataset.upper()
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    kept = corpus.kept_ids()                       # FAISS numbers vectors 0..n-1 in insertion order
    print(f"  exporting {name}: {corpus.n} x {corpus.dim} -> {d}", flush=True)
    with open(os.path.join(d, "base.fvecs"), "wb") as f:
        for _, x in corpus.chunks():
            write_vecs(f, x, np.float32)
    # R-calibration queries: the last quarter (at most 500) tunes DARTH's prediction intervals,
    # the rest trains its predictor. Same query budget as the other methods.
    cq, cg = spec["calib_r"], g["r"]
    n_val = min(500, len(cq) // 4)
    splits = {"train": (cq[:-n_val], cg[:-n_val]), "validation": (cq[-n_val:], cg[-n_val:]),
              "test": (spec["test_q"], g["test"])}
    for split, (q, ids) in splits.items():
        q = normalize(q)
        local = ids if corpus.keep is None else np.searchsorted(kept, ids)
        dist = np.empty(ids.shape, dtype=np.float32)
        for a in range(0, len(q), 256):
            b = min(a + 256, len(q))
            uniq, inv = np.unique(ids[a:b].ravel(), return_inverse=True)   # h5py: increasing, unique rows
            nb = corpus.rows(uniq)[inv].reshape(b - a, ids.shape[1], -1)
            dist[a:b] = ((nb - q[a:b, None, :]) ** 2).sum(axis=2)
        with open(os.path.join(d, f"{split}.fvecs"), "wb") as f:
            write_vecs(f, q, np.float32)
        with open(os.path.join(d, f"{split}.gt.ivecs"), "wb") as f:
            write_vecs(f, local.astype(np.int32), np.int32)
        with open(os.path.join(d, f"{split}.gtd.fvecs"), "wb") as f:
            write_vecs(f, dist, np.float32)
        print(f"    {split}: {len(q)} queries, top-{ids.shape[1]}", flush=True)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(dict(dataset=args.dataset, name=name, n=int(corpus.n), dim=int(corpus.dim), k=int(K),
                       n_train=len(splits["train"][0]), n_val=n_val, n_test=len(splits["test"][0])), f, indent=1)
    print(f"  done: {d}")


def main():
    if args.export_darth:
        return export_darth(load_dataset(args.dataset), args.export_darth)
    t_all = time.time()
    timings = {}
    print("═" * 90)
    print(f"  Unified benchmark: {args.dataset}{'  [SMOKE]' if args.smoke else ''}   settings={SETTINGS}")
    print("═" * 90)
    spec = load_dataset(args.dataset)
    if args.k is not None:
        spec["k"] = args.k
    corpus, K = spec["corpus"], spec["k"]
    test_q, calib_r = spec["test_q"], spec["calib_r"]
    if args.smoke:
        test_q, calib_r = test_q[:N_TEST_SMOKE], calib_r[:N_CALIB_R]
    grid = ef_grid(K)
    cache = os.path.join("unified_cache", spec["tag"] + ("_smoke" if args.smoke else ""))
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(spec["index_path"]) or ".", exist_ok=True)
    print(f"  corpus {corpus.n} x {corpus.dim} (of {corpus.n_total} rows) | K={K} | test {len(test_q)} | "
          f"R-calib {len(calib_r)} | P-calib {N_CALIB_P}")
    print(f"  {spec['notes']}")
    rng = np.random.default_rng(SEED + 1)

    # 1. statistics, KS pool, P-calibration rows (one streaming pass)
    t0 = time.time()
    stats_npz = os.path.join(cache, f"stats{SEED_TAG}.npz")            # P points and KS pool depend on the seed
    stats_bin = os.path.join(cache, f"ada_estimator{SEED_TAG}.bin")
    if os.path.exists(stats_npz):
        st = np.load(stats_npz)
        mean, cov, n_seen, pool, calib_p, p_ids = st["mean"], st["cov"], int(st["n"]), st["pool"], st["calib_p"], st["p_ids"]
    else:
        kept = corpus.kept_ids()
        pool_ids = kept[rng.choice(len(kept), min(KS_POOL, len(kept)), replace=False)]
        p_ids = kept[rng.choice(len(kept), N_CALIB_P, replace=False)]
        del kept
        print("\n[1] streaming pass: mean, covariance, KS pool, P-calibration rows ...", flush=True)
        mean, cov, n_seen, pool, calib_p = stats_pass(corpus, pool_ids, p_ids)
        np.savez(stats_npz, mean=mean, cov=cov, n=n_seen, pool=pool, calib_p=calib_p, p_ids=p_ids)
        # Ada-ef's offline statistics (mean + covariance); the same pass samples the KS pool, a small extra
        save_offline_time(cache, "ada_statistics_s", time.time() - t0)
    assert n_seen == corpus.n, f"stats saw {n_seen} rows, corpus has {corpus.n}"
    write_ada_stats(stats_bin, mean, cov)
    timings["stats"] = time.time() - t0
    ks_mean, ks_vals = ks_fit(pool, mean, cov, test_q, np.random.default_rng(SEED + 2))
    print(f"  KS (score vs Ada-ef's CLT Normal, {len(ks_vals)} test queries): {ks_mean:.4f}")

    # 2. ground truth for test, R-calibration and P-calibration queries (one pass)
    t0 = time.time()
    gt_path = os.path.join(cache, f"gt_k{K}{SEED_TAG}.npz")            # the query split depends on the seed
    bigger = sorted((int(re.search(r"gt_k(\d+)\.npz$", f).group(1)), f)
                    for f in glob.glob(os.path.join(cache, "gt_k*.npz")) if re.search(r"gt_k(\d+)\.npz$", f))
    bigger = [f for kk, f in bigger if kk > K] if not SEED_TAG else []   # untagged files hold seed 42's queries
    if os.path.exists(gt_path):
        g = np.load(gt_path)
        test_gt, r_gt, p_gt = g["test"], g["r"], g["p"]
    elif bigger:
        g = np.load(bigger[0])
        test_gt, r_gt, p_gt = g["test"][:, :K], g["r"][:, :K], g["p"][:, :K]
        print(f"  ground truth: top-{K} taken from {os.path.basename(bigger[0])}")
    else:
        print(f"\n[2] streaming ground truth, top-{K} for {len(test_q) + len(calib_r) + len(calib_p)} queries ...", flush=True)
        allq = np.concatenate([test_q, calib_r, calib_p])
        gt = ground_truth_pass(corpus, allq, K)
        test_gt, r_gt, p_gt = gt[:len(test_q)], gt[len(test_q):len(test_q) + len(calib_r)], gt[len(test_q) + len(calib_r):]
        np.savez(gt_path, test=test_gt, r=r_gt, p=p_gt)
    timings["ground_truth"] = time.time() - t0

    # 3. cluster bins for every K (centroids fit on the KS pool, distances streamed)
    t0 = time.time()
    bins_path = os.path.join(cache, f"cluster_bins_{'_'.join(map(str, K_SWEEP))}{SEED_TAG}.pkl")
    if os.path.exists(bins_path):
        with open(bins_path, "rb") as f:
            centroids, bins = pickle.load(f)
    else:
        print(f"\n[3] cluster bins for K_clusters={K_SWEEP} ...", flush=True)
        centroids, km_secs = {}, {}
        for kc in K_SWEEP:
            tk = time.time()
            if kc == 1:
                centroids[kc] = mean.astype(np.float32).reshape(1, -1)   # the corpus mean, from step 1
            else:
                km = MiniBatchKMeans(n_clusters=kc, random_state=SEED, n_init=3, batch_size=4096).fit(pool)
                centroids[kc] = km.cluster_centers_.astype(np.float32)
            km_secs[kc] = time.time() - tk
        bins, bin_secs = cluster_pass(corpus, centroids)
        with open(bins_path, "wb") as f:
            pickle.dump((centroids, bins), f)
        for kc in K_SWEEP:
            # excludes the shared corpus read; K=1 also needs the corpus mean (Ada-ef's statistics pass)
            save_offline_time(cache, f"ours_bins_K{kc}_s", km_secs[kc] + bin_secs[kc])
    timings["cluster_bins"] = time.time() - t0

    # 4. Ada-ef estimator from streamed statistics; cross-check against the original construction
    verify = None
    if not hasattr(hnsw, "PercEFConfig") or not hasattr(hnsw.AdaEfPaperScorer, "from_stats_file"):
        sys.exit("The C++ extension predates AdaEfPaperScorer.from_stats_file. Rebuild it cleanly:\n"
                 "  cd chao_hybrid_ada_ef && rm -rf build chao_hybrid_ada_ef_cpp*.so && "
                 "python3 setup.py build_ext --inplace && cd ..")
    scorer = hnsw.AdaEfPaperScorer.from_stats_file(stats_bin, QUANTILE_STEP)
    scorer_full = None
    if corpus.n * corpus.dim * 4 <= VERIFY_MAX_BYTES:
        print("\n[4] building Ada-ef's estimator the original way too (corpus fits in RAM) ...", flush=True)
        full = np.concatenate([x for _, x in corpus.chunks()])
        scorer_full = hnsw.AdaEfPaperScorer(full, QUANTILE_STEP)
        del full

    # 5. index: reuse (verified) or build chunk by chunk
    t0 = time.time()
    idx = hnsw.Index(space="l2", dim=corpus.dim)
    if os.path.exists(spec["index_path"]):
        print(f"\n[5] loading index {spec['index_path']} ...", flush=True)
        idx.load_index(spec["index_path"], max_elements=corpus.n)
        ok = (idx.element_count == corpus.n and idx.M == M and idx.ef_construction == EF_CONSTRUCTION)
        print(f"  elements {idx.element_count}, M {idx.M}, ef_construction {idx.ef_construction} -> "
              f"{'matches protocol' if ok else 'DOES NOT MATCH'}")
        if not ok:
            sys.exit("Existing index does not match the frozen protocol; delete or move it and rerun to rebuild.")
    else:
        print(f"\n[5] building index (M={M}, ef_construction={EF_CONSTRUCTION}) chunk by chunk ...", flush=True)
        idx.init_index(max_elements=corpus.n, ef_construction=EF_CONSTRUCTION, M=M)
        done = 0
        for ids, x in corpus.chunks():
            idx.add_items(x, ids)
            done += len(ids)
            if done % 1_000_000 < CHUNK:
                print(f"    ... {done}/{corpus.n} ({time.time() - t0:.0f}s)", flush=True)
        idx.save_index(spec["index_path"])
    timings["index"] = time.time() - t0

    if scorer_full is not None:
        a = [idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer, None)[2] for q in test_q[:50]]
        b = [idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer_full, None)[2] for q in test_q[:50]]
        diff = np.abs(np.asarray(a) - np.asarray(b))
        same = int(np.sum(diff < 1e-3))
        verify = dict(queries=len(diff), identical=same, max_abs_diff=float(diff.max()))
        # A distance sitting on a bin edge can flip one bin between float64 and float32 statistics,
        # so a few differing scores are expected; most must be identical.
        print(f"  Ada-ef score check, streamed vs original statistics: {same}/{len(diff)} queries identical, "
              f"max |diff| {diff.max():.3g} {'(OK)' if same >= 0.9 * len(diff) else '(MISMATCH: investigate before trusting Ada-ef numbers)'}")
        del scorer_full

    # 6. fixed ef (shared by both settings)
    print(f"\n[6] fixed ef on {len(test_q)} test queries ...", flush=True)
    for q in test_q[:500]:                     # warm-up, so the first timed method is not penalised
        idx.search_knn_adaptive(q, K, idx.entry_point, idx.max_level, fixed_efs(K)[2])
    fixed_rows, per_query, fixed_searchers = [], {}, {}
    for ef in fixed_efs(K):
        fsearch = lambda i, q, ef=ef: (idx.search_knn_adaptive(q, K, idx.entry_point, idx.max_level, ef)[0], ef)
        rec, efs, dcs, lat = run_queries(idx, test_q, test_gt, K, fsearch)
        row, pq = summarize(f"Fixed(ef={ef})", rec, efs, dcs, 0, lat)
        fixed_searchers[row["name"]] = fsearch
        fixed_rows.append(row); per_query[row["name"]] = pq
        print(f"  {row['name']:<16} R={row['mean_r']:.4f} p1={row['p1']:.3f} DC={row['total_dc']:.0f} "
              f"lat={row['mean_lat_us']:.0f}us", flush=True)
    if fixed_rows[-1]["mean_r"] < 0.5:
        sys.exit(f"Fixed(ef={fixed_efs(K)[-1]}) reaches only {fixed_rows[-1]['mean_r']:.3f} recall: index labels do not "
                 f"match the ground truth's row ids (wrong index file, or it was built from a different row order).")

    meta = dict(dataset=args.dataset, tag=spec["tag"], smoke=args.smoke, notes=spec["notes"],
                n_corpus=corpus.n, n_total_rows=corpus.n_total, dim=corpus.dim, K=K, n_test=len(test_q),
                n_calib_R=len(calib_r), n_calib_P=N_CALIB_P, target_recall=TARGET_RECALL, ef_cap=EF_CAP,
                M=M, ef_construction=EF_CONSTRUCTION, probe_count=PROBE_COUNT, num_bins=NUM_BINS,
                quantile_step=QUANTILE_STEP, statics_length=STATICS_LENGTH, k_sweep=K_SWEEP,
                default_config=DEFAULT_CONFIG, lat_rounds=args.lat_rounds, ks_mean=ks_mean, ks_per_query=ks_vals,
                ada_stats_verify=verify, index_path=spec["index_path"], reused_index=spec["reuse"])
    try:
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass

    # 7. each calibration setting
    for setting in SETTINGS:
        t_set = time.time()
        calib_q, calib_gt = (calib_p, p_gt) if setting == "P" else (calib_r, r_gt)
        print(f"\n{'═' * 90}\n  Setting {setting}: calibration on {len(calib_q)} "
              f"{'corpus points (paper protocol)' if setting == 'P' else 'held-out real queries'}\n{'═' * 90}")
        rows, pq_setting = list(fixed_rows), dict(per_query)
        searchers = dict(fixed_searchers)          # what the latency pass times

        mef_path = os.path.join(cache, f"calib_min_ef_{setting}{CACHE_TAG}.npz")
        if os.path.exists(mef_path):
            z = np.load(mef_path); calib_min_ef, capped = z["min_ef"], z["capped"]
        else:
            print("  true min-ef per calibration query ...", flush=True)
            tm = time.time()
            calib_min_ef, capped = min_ef_sweep(idx, calib_q, calib_gt, K, grid)
            np.savez(mef_path, min_ef=calib_min_ef, capped=capped)
            save_offline_time(cache, f"ours_min_ef_{setting}_s", time.time() - tm)
        print(f"  calib min-ef: median {np.median(calib_min_ef):.0f}, P90 {np.percentile(calib_min_ef, 90):.0f}, "
              f"capped {capped.mean() * 100:.1f}%")

        # Ada-ef
        ta = time.time()
        ada_scores = np.array([idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer, None)[2] for q in calib_q])
        ada_score_s = time.time() - ta
        rho_ada = finite_or_none(spearmanr(ada_scores, calib_min_ef)[0])
        tab_path = os.path.join(cache, f"ada_table_{setting}{CACHE_TAG}.json")
        if os.path.exists(tab_path):
            with open(tab_path) as f:
                z = json.load(f); ada_table, wae = {int(k): v for k, v in z["table"].items()}, z["wae"]
        else:
            print("  Ada-ef ef-estimation table (group-average probing) ...", flush=True)
            ta = time.time()
            ada_table, wae = ada_target_recall_table(idx, np.round(ada_scores).astype(int), calib_q, calib_gt, K, grid)
            save_offline_time(cache, f"ada_table_{setting}_s", time.time() - ta)
            with open(tab_path, "w") as f:
                json.dump(dict(table=ada_table, wae=wae), f, indent=1)
        for variant, table in (("as shipped", ada_table), ("WAE floor", {s: max(e, wae) for s, e in ada_table.items()})):
            sketch = hnsw.AdaEfPaperSketch([(s, [(e, float(TARGET_RECALL))]) for s, e in table.items()], TARGET_RECALL)
            def ada_search(i, q, sketch=sketch):
                labs, _, _, ef_used = idx.adaptive_search_knn_paper(q, K, STATICS_LENGTH, scorer, sketch)
                return labs, ef_used
            rec, efs, dcs, lat = run_queries(idx, test_q, test_gt, K, ada_search)
            row, pq = summarize(f"Ada-ef ({variant})", rec, efs, dcs, 0, lat, dict(wae=wae))
            searchers[row["name"]] = ada_search
            rows.append(row); pq_setting[row["name"]] = pq
            print(f"  {row['name']:<24} R={row['mean_r']:.4f} p1={row['p1']:.3f} hit={row['pct_target']:.1f}% "
                  f"DC={row['total_dc']:.0f} lat={row['mean_lat_us']:.0f}us", flush=True)

        # Ours
        rho_ours, ours_calib_s, ours_scores_k1, ours_table_len = {}, {}, None, {}
        for kc in K_SWEEP:
            to = time.time()
            near = np.argmin(cdist(calib_q, centroids[kc], metric="sqeuclidean"), axis=1)
            sc = np.array([idx.get_dynamic_probe_score_weighted(calib_q[i], bins[kc][near[i]].tolist(), BIN_WEIGHTS,
                                                                PROBE_COUNT) for i in range(len(calib_q))], dtype=np.float32)
            rho_ours[kc] = finite_or_none(spearmanr(sc, calib_min_ef)[0])
            si = np.round(sc).astype(int)
            tables = {"Isotonic": build_isotonic(si, calib_min_ef, K)}
            ours_calib_s[kc] = time.time() - to          # scoring + Isotonic fit (the default recipe)
            ours_table_len[kc] = len(tables["Isotonic"])
            if kc == 1:
                ours_scores_k1 = sc
            for how in ("Mean", "P90", "P70"):
                tables[how] = table_to_list(build_bucket(si, calib_min_ef, how), K)
            cents, bin_lists = centroids[kc], [b.tolist() for b in bins[kc]]
            for how in RECIPES:
                ef_list = tables[how]
                # thresholds, weights and ef table held in C++ (one config per centroid), as Ada-ef's sketch is
                cfgs = [hnsw.PercEFConfig(b, BIN_WEIGHTS, ef_list, K, EF_CAP, PROBE_COUNT) for b in bin_lists]
                def ours_search(i, q, kc=kc, cents=cents, cfgs=cfgs):
                    # the nearest-centroid lookup is part of our per-query work, so it is timed
                    c = 0 if kc == 1 else int(np.argmin(((cents - q) ** 2).sum(axis=1)))
                    labs, _, ef_used = idx.search_percef(q, K, cfgs[c])
                    return labs, ef_used
                rec, efs, dcs, lat = run_queries(idx, test_q, test_gt, K, ours_search)
                row, pq = summarize(f"Ours (K={kc}, {how})", rec, efs, dcs, kc, lat)
                if row["name"] == DEFAULT_CONFIG:
                    searchers[row["name"]] = ours_search
                rows.append(row); pq_setting[row["name"]] = pq
                print(f"  {row['name']:<24} R={row['mean_r']:.4f} p1={row['p1']:.3f} hit={row['pct_target']:.1f}% "
                      f"DC={row['total_dc']:.0f} lat={row['mean_lat_us']:.0f}us", flush=True)
                with open(os.path.join(RESULTS_DIR, f"ef_table_{setting}_k{kc}_{how.lower()}.json"), "w") as f:
                    json.dump(ef_list, f)

        # Candidate fixes (--fixes, updateAsOf071026.md §6). (a) "scaled": the isotonic table maps a score to
        # the average ef* of similar calibration queries and never checks the mean recall that results;
        # one factor a multiplies the whole table, the smallest a whose calibration mean recall reaches the
        # target (bisection on the calibration queries only). (b) "L=<n>": a shorter probe, for small k,
        # where the 100-distance probe is a large share of a whole search; own scores and isotonic table.
        if args.fixes:
            b1 = bins[1][0].tolist()

            def calib_recall(cfg):
                return float(np.mean([recall_of(idx.search_percef(q, K, cfg)[0], calib_gt[i], K)
                                      for i, q in enumerate(calib_q)]))

            def scale_table(table, L):
                sc_t = lambda a: [int(np.clip(round(v * a), K, EF_CAP)) for v in table]
                cfg_of = lambda a: hnsw.PercEFConfig(b1, BIN_WEIGHTS, sc_t(a), K, EF_CAP, L)
                lo, hi = 0.25, 4.0
                if calib_recall(cfg_of(lo)) >= TARGET_RECALL:
                    return lo, sc_t(lo)
                if calib_recall(cfg_of(hi)) < TARGET_RECALL:
                    return hi, sc_t(hi)
                for _ in range(12):                      # ratio 16 -> about 0.1% resolution
                    mid = float(np.sqrt(lo * hi))
                    if calib_recall(cfg_of(mid)) >= TARGET_RECALL:
                        hi = mid
                    else:
                        lo = mid
                return hi, sc_t(hi)

            iso_default = build_isotonic(np.round(ours_scores_k1).astype(int), calib_min_ef, K)
            variants = [("Ours (K=1, Isotonic, scaled)", PROBE_COUNT, iso_default, True)]
            Ls = args.short_probe
            sc_s = np.array([idx.get_dynamic_probe_score_weighted(q, b1, BIN_WEIGHTS, Ls) for q in calib_q],
                            dtype=np.float32)
            iso_s = build_isotonic(np.round(sc_s).astype(int), calib_min_ef, K)
            rho_short = finite_or_none(spearmanr(sc_s, calib_min_ef)[0])
            variants += [(f"Ours (K=1, Isotonic, L={Ls})", Ls, iso_s, False),
                         (f"Ours (K=1, Isotonic, L={Ls}, scaled)", Ls, iso_s, True)]
            for name, L, table, scale in variants:
                alpha = None
                if scale:
                    tf = time.time()
                    alpha, table = scale_table(table, L)
                    save_offline_time(cache, f"fix_scale_{setting}_L{L}_s", time.time() - tf)
                cfg = hnsw.PercEFConfig(b1, BIN_WEIGHTS, table, K, EF_CAP, L)
                def fix_search(i, q, cfg=cfg):
                    labs, _, ef_used = idx.search_percef(q, K, cfg)
                    return labs, ef_used
                rec, efs, dcs, lat = run_queries(idx, test_q, test_gt, K, fix_search)
                row, pq = summarize(name, rec, efs, dcs, 1, lat,
                                    dict(scale=alpha, probe_len=L, rho=rho_short if L != PROBE_COUNT else None))
                searchers[name] = fix_search
                rows.append(row); pq_setting[name] = pq
                print(f"  {name:<34} R={row['mean_r']:.4f} p1={row['p1']:.3f} DC={row['total_dc']:.0f}"
                      f"{'' if alpha is None else f'  scale {alpha:.3f}'}"
                      f"{'' if L == PROBE_COUNT else f'  rho {rho_short}'}", flush=True)
                with open(os.path.join(RESULTS_DIR, f"ef_table_{setting}_{name.split('(')[1].rstrip(')').replace(', ', '_').replace('=', '')}.json"), "w") as f:
                    json.dump(table, f)

        # Probe-length ablation (--ablation): each score also runs with the other's probe length, so
        # the thresholds are the only difference left (paper, \S3.3). Ada-ef keeps its own code and
        # group-average table; only statics_length changes. Ours keeps its thresholds and isotonic fit.
        ablation_rho = {}
        if args.ablation:
            L_A, L_O = PROBE_COUNT, STATICS_LENGTH
            a_sc = np.array([idx.adaptive_search_knn_paper(q, K, L_A, scorer, None)[2] for q in calib_q])
            ablation_rho[f"ada_L{L_A}"] = finite_or_none(spearmanr(a_sc, calib_min_ef)[0])
            tab_l = os.path.join(cache, f"ada_table_{setting}_L{L_A}{CACHE_TAG}.json")
            if os.path.exists(tab_l):
                with open(tab_l) as f:
                    z = json.load(f); a_table, a_wae = {int(k): v for k, v in z["table"].items()}, z["wae"]
            else:
                print(f"  [ablation] Ada-ef table with a {L_A}-distance probe ...", flush=True)
                a_table, a_wae = ada_target_recall_table(idx, np.round(a_sc).astype(int), calib_q, calib_gt, K, grid)
                with open(tab_l, "w") as f:
                    json.dump(dict(table=a_table, wae=a_wae), f, indent=1)
            for variant, table in (("as shipped", a_table), ("WAE floor", {s_: max(e, a_wae) for s_, e in a_table.items()})):
                sketch = hnsw.AdaEfPaperSketch([(s_, [(e, float(TARGET_RECALL))]) for s_, e in table.items()], TARGET_RECALL)
                def ada_l_search(i, q, sketch=sketch):
                    labs, _, _, ef_used = idx.adaptive_search_knn_paper(q, K, L_A, scorer, sketch)
                    return labs, ef_used
                rec, efs, dcs, lat = run_queries(idx, test_q, test_gt, K, ada_l_search)
                row, pq = summarize(f"Ada-ef L={L_A} ({variant})", rec, efs, dcs, 0, lat, dict(wae=a_wae, probe_len=L_A))
                searchers[row["name"]] = ada_l_search
                rows.append(row); pq_setting[row["name"]] = pq
                print(f"  {row['name']:<30} R={row['mean_r']:.4f} p1={row['p1']:.3f} DC={row['total_dc']:.0f}", flush=True)
            b1 = bins[1][0].tolist()
            o_sc = np.array([idx.get_dynamic_probe_score_weighted(q, b1, BIN_WEIGHTS, L_O) for q in calib_q], dtype=np.float32)
            ablation_rho[f"ours_L{L_O}"] = finite_or_none(spearmanr(o_sc, calib_min_ef)[0])
            o_list = build_isotonic(np.round(o_sc).astype(int), calib_min_ef, K)
            o_cfg = hnsw.PercEFConfig(b1, BIN_WEIGHTS, o_list, K, EF_CAP, L_O)
            def ours_l_search(i, q, o_cfg=o_cfg):
                labs, _, ef_used = idx.search_percef(q, K, o_cfg)
                return labs, ef_used
            rec, efs, dcs, lat = run_queries(idx, test_q, test_gt, K, ours_l_search)
            row, pq = summarize(f"Ours (K=1, Isotonic, L={L_O})", rec, efs, dcs, 1, lat, dict(probe_len=L_O))
            searchers[row["name"]] = ours_l_search
            rows.append(row); pq_setting[row["name"]] = pq
            print(f"  {row['name']:<30} R={row['mean_r']:.4f} p1={row['p1']:.3f} DC={row['total_dc']:.0f}", flush=True)
            print(f"  [ablation] rho: Ada-ef L={L_A} {ablation_rho[f'ada_L{L_A}']} (L={STATICS_LENGTH}: {rho_ada}) | "
                  f"ours L={L_O} {ablation_rho[f'ours_L{L_O}']} (L={PROBE_COUNT}: {rho_ours.get(1)})", flush=True)

        # PercEF's score with Ada-ef's ef table (--group-table): per rounded score, the smallest ef at
        # which the calibration queries with that score reach the target on *average* recall, built by
        # the same function as Ada-ef's table; WAE floor variant as for Ada-ef. Tests whether Ada-ef's
        # tail edge on near-Gaussian data comes from its table (updateAsOf061026.md \S2).
        if args.group_table:
            g_path = os.path.join(cache, f"percef_group_table_{setting}{CACHE_TAG}.json")
            if os.path.exists(g_path):
                with open(g_path) as f:
                    z = json.load(f); g_table, g_wae = {int(k): v for k, v in z["table"].items()}, z["wae"]
            else:
                print("  [group table] PercEF score with Ada-ef's group-average table ...", flush=True)
                g_table, g_wae = ada_target_recall_table(idx, np.round(ours_scores_k1).astype(int), calib_q, calib_gt, K, grid)
                with open(g_path, "w") as f:
                    json.dump(dict(table=g_table, wae=g_wae), f, indent=1)
            b1g = bins[1][0].tolist()
            for variant, table in (("as shipped", g_table), ("WAE floor", {s_: max(e, g_wae) for s_, e in g_table.items()})):
                g_cfg = hnsw.PercEFConfig(b1g, BIN_WEIGHTS, table_to_list(table, K), K, EF_CAP, PROBE_COUNT)
                def g_search(i, q, g_cfg=g_cfg):
                    labs, _, ef_used = idx.search_percef(q, K, g_cfg)
                    return labs, ef_used
                rec, efs, dcs, lat = run_queries(idx, test_q, test_gt, K, g_search)
                row, pq = summarize(f"PercEF + group table ({variant})", rec, efs, dcs, 1, lat, dict(wae=g_wae))
                searchers[row["name"]] = g_search
                rows.append(row); pq_setting[row["name"]] = pq
                print(f"  {row['name']:<34} R={row['mean_r']:.4f} p1={row['p1']:.3f} DC={row['total_dc']:.0f}", flush=True)

        # Latency pass: fixed-ef grid and headline methods; replaces their single-pass latency
        # time only the fixed-ef points that bracket a headline method's recall (all the scorecard needs)
        fx_sorted = sorted(fixed_rows, key=lambda r: r["avg_ef"])
        keep = set()
        for h in (r for r in rows if r["name"] in searchers and not r["name"].startswith("Fixed")):
            for a, b in zip(fx_sorted, fx_sorted[1:]):
                if a["mean_r"] < h["mean_r"] <= b["mean_r"]:
                    keep.update((a["name"], b["name"]))
        searchers = {n: f for n, f in searchers.items() if not n.startswith("Fixed") or n in keep}
        print(f"  latency pass: {len(searchers)} methods x {len(test_q)} queries x {args.lat_rounds} rounds ...", flush=True)
        med = timing_pass(searchers, test_q, args.lat_rounds)
        for j, r in enumerate(rows):
            if r["name"] in med:
                rows[j], pq_setting[r["name"]] = set_latency(r, pq_setting[r["name"]], med[r["name"]])
        for r in rows:
            if r["name"] in med and not r["name"].startswith("Fixed"):
                print(f"  {r['name']:<24} lat={r['mean_lat_us']:.0f}us (median of {args.lat_rounds} rounds)", flush=True)

        # Calibration-time choice between the two scores (discussion only, PAPER_PLAN.md): keep ours
        # (K=1, Isotonic) if its calibration |rho| is at least Ada-ef's; an undefined rho counts as 0.
        # Uses the calibration queries only, so it is a legitimate method; the rows copy the chosen one.
        absr = lambda v: 0.0 if v is None else abs(v)
        chose_ours = absr(rho_ours.get(1)) >= absr(rho_ada)
        for variant in ("as shipped", "WAE floor"):
            src = DEFAULT_CONFIG if chose_ours else f"Ada-ef ({variant})"
            row = dict(next(r for r in rows if r["name"] == src))
            row.update(name=f"Choice (with Ada-ef {variant})", chosen=src)
            rows.append(row); pq_setting[row["name"]] = pq_setting[src]

        # Summary against Ada-ef (as shipped)
        ada = next(r for r in rows if r["name"] == "Ada-ef (as shipped)")
        ours_rows = [r for r in rows if r["name"].startswith("Ours")]
        fixed = [r for r in rows if r["name"].startswith("Fixed")]
        default = next(r for r in rows if r["name"] == DEFAULT_CONFIG)
        pct = lambda x: None if x is None else (x["dc"] - ada["total_dc"]) / ada["total_dc"] * 100
        eq_r, eq_t = dc_at_quality(ours_rows, "mean_r", ada["mean_r"]), dc_at_quality(ours_rows, "pct_target", ada["pct_target"])
        fx_dc = fixed_interp(fixed, ada["mean_r"], "total_dc")
        fx_r = None if fx_dc is None else dict(dc=fx_dc, bound=False, via="fixed ef, log-log interpolation")
        # Scorecard against a tuned fixed ef at the same mean recall (DC and latency savings, tail gains)
        main_names = ["Ada-ef (as shipped)", "Ada-ef (WAE floor)", DEFAULT_CONFIG,
                      "Choice (with Ada-ef as shipped)", "Choice (with Ada-ef WAE floor)"]
        if args.ablation:
            main_names += [f"Ada-ef L={PROBE_COUNT} (as shipped)", f"Ada-ef L={PROBE_COUNT} (WAE floor)",
                           f"Ours (K=1, Isotonic, L={STATICS_LENGTH})"]
        if args.group_table:
            main_names += ["PercEF + group table (as shipped)", "PercEF + group table (WAE floor)"]
        if args.fixes:
            main_names += ["Ours (K=1, Isotonic, scaled)", f"Ours (K=1, Isotonic, L={args.short_probe})",
                           f"Ours (K=1, Isotonic, L={args.short_probe}, scaled)"]
        scorecard = {n: vs_fixed(next(r for r in rows if r["name"] == n), fixed) for n in main_names}
        # Calibration diagnostics (the factors of updateAsOf280926.md §9)
        ada_by_score = np.array([ada_table.get(int(s), wae) for s in np.round(ada_scores).astype(int)])
        diagnostics = dict(
            floor_share=float(np.mean(calib_min_ef <= grid[0])), capped_share=float(capped.mean()),
            headroom=float(np.percentile(calib_min_ef, 90) / max(np.mean(np.maximum(calib_min_ef, K)), 1e-9)),
            min_ef_cv=float(np.std(calib_min_ef) / max(np.mean(calib_min_ef), 1e-9)),
            ada_distinct_scores=int(len(np.unique(np.round(ada_scores)))),
            ada_distinct_table_efs=int(len(np.unique(ada_by_score))),
            ours_k1_distinct_scores=None if ours_scores_k1 is None else int(len(np.unique(np.round(ours_scores_k1)))))
        # Offline cost: seconds (first-run times, kept with the cache) and bytes each method keeps
        ot = load_offline_times(cache)
        d = corpus.dim
        offline = dict(
            ada=dict(statistics_s=ot.get("ada_statistics_s"), score_calib_s=ada_score_s,
                     table_s=ot.get(f"ada_table_{setting}_s"),
                     memory_bytes=int(d * d * 4 + 2 * d * 4 + len(ada_table) * 8)),
            ours={f"K={kc}": dict(bins_s=ot.get(f"ours_bins_K{kc}_s"), min_ef_s=ot.get(f"ours_min_ef_{setting}_s"),
                                  score_fit_s=ours_calib_s[kc],
                                  memory_bytes=int(kc * NUM_BINS * 4 + kc * d * 4 + ours_table_len[kc] * 4))
                  for kc in K_SWEEP},
            note="statistics_s is the streaming mean+covariance pass (it also samples the KS pool); "
                 "bins_s excludes the shared corpus read; K=1 also uses the corpus mean from that pass. "
                 "None = computed by an earlier run, before timing was recorded.")
        summary = dict(setting=setting, n_calib=len(calib_q), rho_ada=rho_ada, rho_ours=rho_ours,
                       choice="ours" if chose_ours else "Ada-ef", scorecard_vs_fixed=scorecard,
                       ablation_rho=ablation_rho or None,
                       diagnostics=diagnostics, offline=offline,
                       calib_min_ef_capped_frac=float(capped.mean()),
                       ada=ada, default=default,
                       ours_dc_at_ada_recall=eq_r, ours_dc_at_ada_recall_pct=pct(eq_r),
                       ours_dc_at_ada_target_hit=eq_t, ours_dc_at_ada_target_hit_pct=pct(eq_t),
                       fixed_dc_at_ada_recall=fx_r, fixed_dc_at_ada_recall_pct=pct(fx_r),
                       fixed_p1_at_ada_recall=fixed_interp(fixed, ada["mean_r"], "p1"),
                       fixed_p5_at_ada_recall=fixed_interp(fixed, ada["mean_r"], "p5"))
        with open(os.path.join(RESULTS_DIR, f"rows_{setting}.json"), "w") as f:
            json.dump(rows, f, indent=1)
        with open(os.path.join(RESULTS_DIR, f"summary_{setting}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        np.savez_compressed(os.path.join(RESULTS_DIR, f"per_query_{setting}.npz"),
                            **{f"{n}|{k}": v for n, d in pq_setting.items() for k, v in d.items()})
        timings[f"setting_{setting}"] = time.time() - t_set

        print(f"\n  SUMMARY {args.dataset} / setting {setting}   (DC includes our probe; target {TARGET_RECALL})")
        print(f"  {'method':<24} {'meanR':>7} {'p1':>6} {'p5':>6} {'hit%':>6} {'DC':>9} {'avg ef':>7}")
        for r in [ada, next(r for r in rows if r['name'] == 'Ada-ef (WAE floor)'), default]:
            print(f"  {r['name']:<24} {r['mean_r']:>7.4f} {r['p1']:>6.3f} {r['p5']:>6.3f} {r['pct_target']:>6.1f} "
                  f"{r['total_dc']:>9.0f} {r['avg_ef']:>7.0f}")
        f1 = lambda v: "n/a" if v is None else f"{v:+.1f}%"
        fr = lambda v: "undefined (constant scores)" if v is None else f"{v:+.3f}"
        print(f"  rho: Ada-ef {fr(rho_ada)} | ours " + ", ".join(f"K={k} {fr(v)}" for k, v in rho_ours.items()))
        print(f"  at Ada-ef's mean recall {ada['mean_r']:.4f}: ours frontier {f1(pct(eq_r))}"
              f"{' (bound)' if eq_r and eq_r['bound'] else ''} | fixed ef {f1(pct(fx_r))}"
              f"{' (bound)' if fx_r and fx_r['bound'] else ''}")
        if summary["fixed_p1_at_ada_recall"] is not None:
            print(f"  tail at that recall: Ada-ef p1 {ada['p1']:.3f} / fixed ef p1 {summary['fixed_p1_at_ada_recall']:.3f}")
        fp = lambda v, f="{:+.1f}%": "n/a" if v is None else f.format(v)
        print(f"\n  SCORECARD vs a tuned fixed ef at the same mean recall   (choice picked {summary['choice']})")
        print(f"  {'method':<32} {'meanR':>7} {'save DC':>8} {'save lat':>9} {'p1 gain':>8} {'p5 gain':>8} {'lat us':>7}")
        for n, sc in scorecard.items():
            r = next(x for x in rows if x["name"] == n)
            print(f"  {n:<32} {r['mean_r']:>7.4f} {fp(sc['saving_dc_pct']):>8} {fp(sc['saving_lat_pct']):>9} "
                  f"{fp(sc['p1_gain'], '{:+.3f}'):>8} {fp(sc['p5_gain'], '{:+.3f}'):>8} {r['mean_lat_us']:>7.0f}"
                  f"{'' if sc['adapts'] else '   (one ef for all queries: no adaptation)'}")
        dg = diagnostics
        print(f"  diagnostics: floor {dg['floor_share'] * 100:.1f}%, capped {dg['capped_share'] * 100:.1f}%, "
              f"headroom {dg['headroom']:.2f}, Ada-ef distinct scores {dg['ada_distinct_scores']} "
              f"(table efs {dg['ada_distinct_table_efs']}), ours K=1 distinct scores {dg['ours_k1_distinct_scores']}")
        o = offline
        print(f"  offline: Ada-ef stats {fp(o['ada']['statistics_s'], '{:.0f}s')}, table {fp(o['ada']['table_s'], '{:.0f}s')}, "
              f"{o['ada']['memory_bytes'] / 1e6:.2f} MB | ours K=1 bins {fp(o['ours']['K=1']['bins_s'], '{:.0f}s')}, "
              f"min-ef {fp(o['ours']['K=1']['min_ef_s'], '{:.0f}s')}, {o['ours']['K=1']['memory_bytes'] / 1e3:.1f} KB")

    meta["timings_s"] = timings
    meta["total_s"] = time.time() - t_all
    with open(os.path.join(RESULTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nDone in {meta['total_s'] / 60:.1f} min. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
