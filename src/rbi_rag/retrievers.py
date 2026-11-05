"""Retrievers. Interface: retriever.retrieve(query, k, qid=None) -> list[dict] (chunk + 'score').

Aage ke steps (BM25, vector, hybrid, reranker ...) isi REGISTRY mein judenge.
"""
from __future__ import annotations

import random
from collections import defaultdict

from rbi_rag.evalr import page_hit
from rbi_rag.golden import build_probe
from rbi_rag.relations import norm


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


REGISTRY = {
    "random": lambda chunks, golden: RandomRetriever(chunks, golden),
    "oracle_page": lambda chunks, golden: OracleRetriever(chunks, golden, "page"),
    "oracle_quote": lambda chunks, golden: OracleRetriever(chunks, golden, "quote"),
    "idf_overlap": lambda chunks, golden: IdfOverlapRetriever(chunks, golden),
}


def build(name, chunks, golden):
    if name not in REGISTRY:
        raise KeyError(f"unknown retriever '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](chunks, golden)