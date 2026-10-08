#!/usr/bin/env python3
"""
Download the raw files benchmark_unified.py needs for Cohere-1024 and LAION-I2I,
from exactly the sources the Ada-ef authors use (experiments_driver/data_prep.ipynb
in github.com/chaozhang-cs/hnsw-ada-ef):

  Cohere-1024 (MS MARCO V2.1, Cohere embed-english-v3, 1024-d, float16):
    https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main/
      passages_npy/msmarco_v2.1_doc_segmented_{00..09}.npy   (the paper's 18,380,565 passages)
      queries_jsonl/queries.jsonl.gz                          (1,677 queries, field "emb")
    The full 10 files need ~78 GB of HNSW index in RAM; this server has 62 GB, so by default
    only files 00-04 are pulled (~9.51M passages, 52% of the paper's corpus, ~19.5 GB on disk,
    ~40 GB of index). Stated as a subset in the paper.

  LAION-I2I (CLIP image embeddings, 512-d, float16, 1,000,448 rows per shard):
    https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_{i}.npy
    The paper uses all 31 shards (~67 GB of index); by default shards 0-19 (~20M rows, ~44 GB of
    index). Shards already in laion_i2i_subset/shards/ are kept, not re-downloaded.

Nothing is concatenated or converted: benchmark_unified.py streams these files directly from
disk. Downloads resume from a .part file if interrupted.

Usage (on the server):
  python3 download_unified_data.py                     # Cohere 00-04 + LAION 0-19
  python3 download_unified_data.py --cohere-files 10   # paper-exact Cohere (needs a bigger machine)
  python3 download_unified_data.py --laion-shards 31   # paper-exact LAION (needs a bigger machine)
  python3 download_unified_data.py --only cohere
"""

import os
import argparse
import requests
from tqdm import tqdm

COHERE_BASE = "https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3/resolve/main"
COHERE_DIR = "cohere_msmarco_v21_npy"
LAION_URL = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_{i}.npy"
LAION_SHARD_DIR = os.path.join("laion_i2i_subset", "shards")   # same folder the old LAION script used


def download(url, dest):
    if os.path.exists(dest):
        print(f"  have {dest}")
        return
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
        if done and resp.status_code != 206:      # server ignored the range: start over
            done = 0
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) + done
        with open(part, "ab" if done else "wb") as f, \
                tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 22):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(part, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohere-files", type=int, default=5, help="passage files 00..N-1 (paper: 10)")
    ap.add_argument("--laion-shards", type=int, default=20, help="image shards 0..N-1 (paper: 31)")
    ap.add_argument("--only", choices=["cohere", "laion"], default=None)
    args = ap.parse_args()

    if args.only in (None, "cohere"):
        os.makedirs(COHERE_DIR, exist_ok=True)
        print(f"Cohere: queries + {args.cohere_files} passage file(s) -> {COHERE_DIR}/")
        download(f"{COHERE_BASE}/queries_jsonl/queries.jsonl.gz", os.path.join(COHERE_DIR, "queries.jsonl.gz"))
        for i in range(args.cohere_files):
            name = f"msmarco_v2.1_doc_segmented_{i:02d}.npy"
            download(f"{COHERE_BASE}/passages_npy/{name}", os.path.join(COHERE_DIR, name))

    if args.only in (None, "laion"):
        os.makedirs(LAION_SHARD_DIR, exist_ok=True)
        print(f"LAION: {args.laion_shards} shard(s) -> {LAION_SHARD_DIR}/")
        for i in range(args.laion_shards):
            download(LAION_URL.format(i=i), os.path.join(LAION_SHARD_DIR, f"img_emb_{i}.npy"))

    print("Done.")


if __name__ == "__main__":
    main()
