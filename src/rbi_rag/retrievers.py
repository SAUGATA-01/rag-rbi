"""Retrievers. Interface: retriever.retrieve(query, k, qid=None) -> list[dict] (chunk + 'score').

Aage ke steps (BM25, vector, hybrid, reranker ...) isi REGISTRY mein judenge.
"""
from __future__ import annotations

import random
from collections import defaultdict

from rbi_rag.bm25 import BM25
from rbi_rag.evalr import page_hit
from rbi_rag.golden import build_probe
from rbi_rag.relations import norm
from rbi_rag.textproc import tokenize
from rbi_rag.vector import EmbeddingCache, HashEmbedder, STEmbedder, VectorIndex


class RandomRetriever:
    """Lower bound: seeded random chunks."""
    name = "random"

    def __init__(self, chunks, golden=None, seed=7):
        self.chunks, self.seed = chunks, seed

    def retrieve(self, query, k, qid=None):
        rnd = random.Random(f"{self.seed}:{qid or query}")
        idx = rnd.sample(range(len(self.chunks)), min(k, len(self.chunks)))
        return [dict(self.chunks[i], score=0.0) for i in idx]


class OracleRetriever:
    """Upper bound / harness test: gold evidence wale chunks pehle (CHEATING, sirf eval check ke liye)."""

    def __init__(self, chunks, golden, mode="page"):
        self.mode, self.chunks = mode, chunks
        self.by_file = defaultdict(list)
        for c in chunks:
            self.by_file[c["file"]].append(c)
        self.golden = {g["id"]: g for g in golden}
        self.name = f"oracle_{mode}"

    def retrieve(self, query, k, qid=None):
        rec = self.golden.get(qid) or {}
        out, seen = [], set()
        for e in rec.get("evidence", []):
            for c in self.by_file.get(e["file"], []):
                ok = page_hit(c, e["file"], e["page"]) if self.mode == "page" else norm(e["quote"]) in norm(c["text"])
                if ok and c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    out.append(dict(c, score=1.0))
                    break
        return out[:k]


class IdfOverlapRetriever:
    """Throwaway lexical baseline (Step 6 probe). BM25 NAHI hai, sirf harness ko non-trivial numbers dene ke liye."""
    name = "idf_overlap"

    def __init__(self, chunks, golden=None):
        self.search = build_probe(chunks)

    def retrieve(self, query, k, qid=None):
        return [dict(c, score=s) for s, c in self.search(query, k)]


class BM25Retriever:
    """BM25 scratch se (numpy). field = 'embed_text' (title|heading|clause + text) ya 'text'."""

    def __init__(self, chunks, golden=None, field="embed_text", use_stem=True, k1=1.5, b=0.75, name="bm25"):
        self.name, self.chunks, self.use_stem = name, chunks, use_stem
        self.bm = BM25([tokenize(c[field], use_stem=use_stem) for c in chunks], k1=k1, b=b)

    def retrieve(self, query, k, qid=None):
        return [dict(self.chunks[i], score=s) for i, s in self.bm.top(tokenize(query, use_stem=self.use_stem), k)]


class VectorRetriever:
    """Naive dense retrieval: har chunk ka embed_text embed, query ke saath cosine. Embeddings disk par cache."""

    def __init__(self, chunks, golden=None, embedder=None, cache_root="data/cache/embeddings", name="vector"):
        self.name, self.chunks = name, chunks
        self.embedder = embedder or STEmbedder()
        cache = EmbeddingCache(cache_root, self.embedder.name)
        mat, self.stats = cache.get_matrix([c["embed_text"] for c in chunks], self.embedder)
        self.index = VectorIndex(mat)

    def retrieve(self, query, k, qid=None):
        q = self.embedder.encode_queries([query])[0]
        return [dict(self.chunks[i], score=s) for i, s in self.index.top(q, k)]


REGISTRY = {
    "random": lambda chunks, golden: RandomRetriever(chunks, golden),
    "oracle_page": lambda chunks, golden: OracleRetriever(chunks, golden, "page"),
    "oracle_quote": lambda chunks, golden: OracleRetriever(chunks, golden, "quote"),
    "idf_overlap": lambda chunks, golden: IdfOverlapRetriever(chunks, golden),
    "bm25": lambda chunks, golden: BM25Retriever(chunks, golden, name="bm25"),
    "bm25_text": lambda chunks, golden: BM25Retriever(chunks, golden, field="text", name="bm25_text"),
    "bm25_nostem": lambda chunks, golden: BM25Retriever(chunks, golden, use_stem=False, name="bm25_nostem"),
    "vector_bge_small": lambda chunks, golden: VectorRetriever(chunks, golden, STEmbedder(), name="vector_bge_small"),
    "vector_bge_small_noprefix": lambda chunks, golden: VectorRetriever(
        chunks, golden, STEmbedder(query_prefix=""), name="vector_bge_small_noprefix"),
    "vector_hash": lambda chunks, golden: VectorRetriever(
        chunks, golden, HashEmbedder(), cache_root="data/cache/embeddings_debug", name="vector_hash"),
}


def build(name, chunks, golden):
    if name not in REGISTRY:
        raise KeyError(f"unknown retriever '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](chunks, golden)