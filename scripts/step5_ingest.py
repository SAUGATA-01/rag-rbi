"""Step 5: sources.csv + PDFs -> chunks (incremental) -> corpus.jsonl + manifest + quality report."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rbi_rag.chunker import ChunkConfig, chunk_pages, load_pages
from rbi_rag.ingest import (cache_valid, cfg_key, corpus_manifest, doc_report, enrich,
                            sha256_file)

RAW = Path("data/raw_pdfs")
META = Path("data/metadata")
SOURCES = META / "sources.csv"
STATUS = META / "doc_status.csv"
MANIFEST = META / "corpus_manifest.json"
PROC = Path("data/processed")
CHUNKS = PROC / "chunks"
CACHE = PROC / "ingest_cache"
CORPUS = PROC / "corpus.jsonl"
REPORT = Path("eval/results/step5_ingest_report.csv")
REQUIRED_META = ["title", "circular_no", "date"]  # Required for citations


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", default="pymupdf", choices=["pymupdf", "pdfplumber"])
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=350)
    ap.add_argument("--force", action="store_true", help="Ignore cache and rechunk all documents")
    args = ap.parse_args()

    cfg = ChunkConfig(min_words=args.min_words, max_words=args.max_words)
    key = cfg_key(cfg, args.parser)
    for d in (CHUNKS, CACHE, REPORT.parent):
        d.mkdir(parents=True, exist_ok=True)

    sources = {r["file"]: r for r in read_csv(SOURCES) if r.get("file")}
    status = {r["file"]: r for r in read_csv(STATUS)}
    if not sources:
        print("sources.csv is empty or missing (run Step 1).")
        sys.exit(1)
    if not status:
        print("NOTE: doc_status.csv was not found; all document statuses will be 'unknown' (run Step 4).")

    errors, warns = [], []
    on_disk = {p.name for p in RAW.glob("*.pdf")}
    for f in sorted(on_disk - set(sources)):
        warns.append(f"PDF not listed in sources.csv: {f} (run scripts/make_manifest.py)")
    for f in sorted(set(sources) - on_disk):
        errors.append(f"Listed in sources.csv but PDF not found: {f}")
    for f, r in sources.items():
        miss = [k for k in REQUIRED_META if not (r.get(k) or "").strip()]
        if miss:
            warns.append(f"{f}: missing metadata ({', '.join(miss)})")
    dup = [c for c, n in Counter((r.get("circular_no") or "").strip() for r in sources.values()).items()
           if c and n > 1]
    for c in dup:
        warns.append(f"Duplicate circular_no: {c}")

    reports, n_cached, n_new = [], 0, 0
    all_rows = []
    for fname in sorted(set(sources) & on_disk):
        pdf = RAW / fname
        stem = pdf.stem
        cache_f, chunk_f = CACHE / f"{stem}.json", CHUNKS / f"{stem}.jsonl"
        pdf_sha = sha256_file(pdf)
        rec = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else None
        if not args.force and chunk_f.exists() and cache_valid(rec, pdf_sha, key):
            n_cached += 1
            pages_n, empty_n = rec["pages"], rec["empty_pages"]
            with chunk_f.open(encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f if line.strip()]
        else:
            try:
                pages = load_pages(pdf, parser=args.parser)
                meta = dict(sources[fname])
                meta["file"] = fname
                chunks = chunk_pages(pages, stem, meta, cfg)
            except Exception as e:
                errors.append(f"{fname}: parsing/chunking failed: {str(e)[:100]}")
                continue
            pages_n = len(pages)
            empty_n = sum(1 for t in pages if len(t.strip()) < 20)
            with chunk_f.open("w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            cache_f.write_text(json.dumps({"pdf_sha": pdf_sha, "cfg_key": key, "pages": pages_n,
                                           "empty_pages": empty_n}), encoding="utf-8")
            n_new += 1
        rep = doc_report(fname, chunks, pages_n, empty_n)
        rep["type"] = sources[fname].get("type", "")
        rep["status"] = (status.get(fname) or {}).get("status", "unknown")
        reports.append(rep)
        all_rows += [enrich(c, status.get(fname)) for c in chunks]

    if not all_rows:
        print("No chunks were created.")
        for e in errors:
            print("ERROR", e)
        sys.exit(1)

    # corpus.jsonl: a single file with document status (status changes do not require rechunking)
    with CORPUS.open("w", encoding="utf-8") as f:
        for c in all_rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    man = corpus_manifest(CORPUS, len(reports), len(all_rows), args.parser, cfg, SOURCES, STATUS)
    MANIFEST.write_text(json.dumps(man, indent=2), encoding="utf-8")

    fields = ["file", "type", "status", "pages", "chunks", "median_words", "noclause_pct",
              "table_pct", "empty_page_pct", "flags"]
    with REPORT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(reports)

    # ---- summary ----
    print(f"docs: {len(reports)} (cache hits {n_cached}, new/rechunked {n_new}) | chunks: {len(all_rows)}")
    print("by type:   " + " | ".join(f"{k or '?'} {v}" for k, v in sorted(Counter(r['type'] for r in reports).items())))
    print("by status: " + " | ".join(f"{k} {v}" for k, v in sorted(Counter(r['status'] for r in reports).items())))
    groups = defaultdict(set)
    for c in all_rows:
        groups[c["text_hash"]].add(c["file"])
    cross = sum(1 for s in groups.values() if len(s) > 1)
    print(f"Identical chunk text across different documents: {cross} groups (informational; not removed)")
    flagged = [r for r in reports if r["flags"]]
    print(f"\nFlagged documents (review recommended): {len(flagged)}")
    for r in flagged:
        print(f"  {r['file']:<34} {r['flags']}  [pages {r['pages']}, chunks {r['chunks']}, median {r['median_words']} words]")
    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\ncorpus_sha: {man['corpus_sha']}  ->  {MANIFEST}")
    print(f"Report: {REPORT}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()