"""Step 3: raw PDFs -> clause-aware chunks (JSONL) + stats + manual-review sample."""
import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from rbi_rag.chunker import ChunkConfig, chunk_pages, load_pages

RAW = Path("data/raw_pdfs")
SOURCES = Path("data/metadata/sources.csv")
OUT = Path("data/processed/chunks")
STATS = Path("eval/results/step3_chunk_stats.csv")
REVIEW_TXT = OUT / "review_sample.txt"
REVIEW_CSV = Path("eval/results/step3_manual_review.csv")


def read_sources():
    if not SOURCES.exists():
        return {}
    with SOURCES.open(encoding="utf-8-sig", newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f) if r.get("file")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", default="pymupdf", choices=["pymupdf", "pdfplumber"])
    ap.add_argument("--only", default="", help="filename mein ye text ho to hi chalao")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=350)
    ap.add_argument("--show", type=int, default=10, help="review ke liye random chunks")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = ChunkConfig(min_words=args.min_words, max_words=args.max_words)
    sources = read_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    STATS.parent.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in sorted(RAW.glob("*.pdf")) if args.only in p.name]
    if not pdfs:
        print("Koi PDF nahi mili.")
        return

    all_chunks, rows = [], []
    for p in pdfs:
        meta = dict(sources.get(p.name, {}))
        meta["file"] = p.name
        try:
            pages = load_pages(p, parser=args.parser)
            chunks = chunk_pages(pages, p.stem, meta, cfg)
        except Exception as e:
            print(f"[ERROR] {p.name}: {str(e)[:120]}")
            rows.append({"file": p.name, "error": str(e)[:120]})
            continue
        with (OUT / f"{p.stem}.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        words = [c["n_words"] for c in chunks] or [0]
        rows.append({
            "file": p.name, "pages": len(pages), "chunks": len(chunks),
            "median_words": int(statistics.median(words)), "max_words": max(words),
            "pct_below_min": round(100 * sum(w < cfg.min_words for w in words) / len(words), 1),
            "pct_above_max": round(100 * sum(w > cfg.max_words for w in words) / len(words), 1),
            "no_clause_chunks": sum(1 for c in chunks if not c["clause_ids"]),
            "table_flagged": sum(1 for c in chunks if c["maybe_table"]),
            "error": "",
        })
        all_chunks += chunks
        print(f"{p.name}: {len(pages)} pages -> {len(chunks)} chunks", flush=True)

    fields = ["file", "pages", "chunks", "median_words", "max_words", "pct_below_min",
              "pct_above_max", "no_clause_chunks", "table_flagged", "error"]
    with STATS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    good = [r for r in rows if not r.get("error")]
    print("\nfile                              pages chunks med_w max_w <min% >max% noclause tables")
    for r in good:
        print(f"{r['file'][:32]:<32} {r['pages']:>6} {r['chunks']:>6} {r['median_words']:>5} "
              f"{r['max_words']:>5} {r['pct_below_min']:>5} {r['pct_above_max']:>5} "
              f"{r['no_clause_chunks']:>8} {r['table_flagged']:>6}")
    print(f"\nTOTAL: {len(good)} files, {sum(r['chunks'] for r in good)} chunks, "
          f"errors: {len(rows) - len(good)}")

    # ---- manual review sample ----
    if args.show and all_chunks:
        rnd = random.Random(args.seed)
        pick = rnd.sample(all_chunks, min(args.show, len(all_chunks)))
        with REVIEW_TXT.open("w", encoding="utf-8") as f:
            for i, c in enumerate(pick, 1):
                f.write(f"##### {i}. {c['chunk_id']}\n"
                        f"file: {c['file']} | pages {c['page_start']}-{c['page_end']} | "
                        f"clause: {c['clause_ids'] or '-'} | words: {c['n_words']} | "
                        f"table_flag: {c['maybe_table']}\n"
                        f"heading: {c['heading']}\n\n{c['text']}\n\n")
        with REVIEW_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["chunk_id", "boundary_ok", "complete_ok", "page_ok",
                        "noise_free", "table_ok", "notes"])
            for c in pick:
                w.writerow([c["chunk_id"], "", "", "", "", "", ""])
        print(f"Review sample: {REVIEW_TXT}\nScoring sheet: {REVIEW_CSV}")


if __name__ == "__main__":
    main()