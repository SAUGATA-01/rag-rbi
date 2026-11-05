"""Step 8: corpus ke chunks ke embeddings banao / refresh karo (incremental cache).

  python scripts\\step8_embed.py --limit 100     # pehle chhota timing test (cache mein save nahi hota)
  python scripts\\step8_embed.py                 # poora corpus, cache mein save
  python scripts\\step8_embed.py --embedder hash # debug: koi model download nahi
"""
import argparse
import json
import sys
import time
from pathlib import Path

from rbi_rag.vector import EmbeddingCache, HashEmbedder, STEmbedder

CORPUS = Path("data/processed/corpus.jsonl")
ROOT = Path("data/cache/embeddings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["st", "hash"], default="st")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="sirf itne chunks (timing test, cache save nahi)")
    a = ap.parse_args()

    if not CORPUS.exists():
        print("corpus.jsonl nahi mili. Pehle Step 5 chalao.")
        sys.exit(1)
    with CORPUS.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    texts = [c["embed_text"] for c in chunks]

    t0 = time.time()
    emb = (HashEmbedder() if a.embedder == "hash"
           else STEmbedder(a.model, a.max_seq_length, a.batch_size))
    print(f"embedder: {emb.name} (load {time.time() - t0:.1f}s) | corpus chunks: {len(texts)}")

    if a.limit:
        sample = texts[:a.limit]
        t1 = time.time()
        emb.encode_passages(sample)
        dt = time.time() - t1
        rate = len(sample) / dt if dt > 0 else float("inf")
        print(f"timing: {len(sample)} chunks in {dt:.1f}s = {rate:.1f} chunks/s")
        print(f"poore corpus ka andaza: ~{len(texts) / rate / 60:.1f} min (pehli baar; baad mein cache)")
        return

    cache = EmbeddingCache(ROOT / ("debug" if a.embedder == "hash" else ""), emb.name)
    mat, st = cache.get_matrix(texts, emb)
    size = sum(p.stat().st_size for p in cache.dir.glob("*")) / 1e6 if cache.dir.exists() else 0
    print(f"chunks {st['chunks']} | embedded now {st['embedded']} | from cache {st['cached']} | "
          f"{st['seconds']}s | matrix {mat.shape} | cache {size:.1f} MB at {cache.dir}")


if __name__ == "__main__":
    main()