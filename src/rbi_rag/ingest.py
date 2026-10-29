"""Step 5: ingestion helpers (incremental cache, status join, quality flags, corpus manifest)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from rbi_rag import chunker
from rbi_rag.relations import norm

STATUS_FIELDS = ["status", "superseded_by", "amended_by", "withdrawn_by", "latest_active"]

# Shuruaati thresholds: 50 docs ka distribution dekh ke tune karna (ye review flags hain, auto-drop nahi)
THRESH = {
    "noclause_pct": 50.0,
    "table_pct": 30.0,
    "empty_page_pct": 20.0,
    "min_median_words": 25,
    "max_median_words": 300,
    "min_chunks_per_page": 0.5,
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def chunker_sha() -> str:
    """chunker.py ka code badle to cache khud invalid ho jaye."""
    return sha256_bytes(Path(chunker.__file__).read_bytes())[:12]


def cfg_key(cfg, parser: str) -> str:
    payload = json.dumps({"cfg": asdict(cfg), "parser": parser, "chunker": chunker_sha()}, sort_keys=True)
    return sha256_bytes(payload.encode())[:12]


def cache_valid(rec, pdf_sha: str, key: str) -> bool:
    return bool(rec) and rec.get("pdf_sha") == pdf_sha and rec.get("cfg_key") == key


def text_hash(text: str) -> str:
    """Whitespace/punctuation-insensitive hash (duplicate aur 'what changed' diff ke kaam aayega)."""
    return hashlib.sha1(norm(text).encode()).hexdigest()[:16]


def enrich(chunk: dict, status_row: dict | None) -> dict:
    c = dict(chunk)
    for k in STATUS_FIELDS:
        c[k] = (status_row or {}).get(k) or ("unknown" if k == "status" and not status_row else "")
    c["text_hash"] = text_hash(c["text"])
    return c


def doc_report(file: str, chunks: list, pages: int, empty_pages: int, t: dict | None = None) -> dict:
    t = t or THRESH
    n = len(chunks)
    words = [c["n_words"] for c in chunks] or [0]
    noclause = round(100 * sum(1 for c in chunks if not c["clause_ids"]) / max(1, n), 1)
    table = round(100 * sum(1 for c in chunks if c["maybe_table"]) / max(1, n), 1)
    empty = round(100 * empty_pages / max(1, pages), 1)
    med = int(median(words))
    flags = []
    if n == 0:
        flags.append("no_chunks")
    else:
        if noclause > t["noclause_pct"]:
            flags.append("many_noclause")
        if table > t["table_pct"]:
            flags.append("many_tables")
        if med < t["min_median_words"]:
            flags.append("tiny_chunks")
        if med > t["max_median_words"]:
            flags.append("huge_chunks")
        if pages >= 5 and n / pages < t["min_chunks_per_page"]:
            flags.append("few_chunks")
    if empty > t["empty_page_pct"]:
        flags.append("empty_pages")
    return {"file": file, "pages": pages, "chunks": n, "median_words": med,
            "noclause_pct": noclause, "table_pct": table, "empty_page_pct": empty,
            "flags": "|".join(flags)}


def corpus_manifest(corpus_path, n_docs, n_chunks, parser, cfg, sources_path, status_path) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "parser": parser,
        "chunk_config": asdict(cfg),
        "chunker_sha": chunker_sha(),
        "sources_sha": sha256_file(sources_path)[:12],
        "status_sha": sha256_file(status_path)[:12] if Path(status_path).exists() else "",
        "corpus_sha": sha256_file(corpus_path)[:12],
    }