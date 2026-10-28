
"""Step 4: validate relations.csv and derive doc_status.csv."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from rbi_rag.chunker import load_pages
from rbi_rag.relations import REL_FIELDS, derive_status, validate

META = Path("data/metadata")
SOURCES = META / "sources.csv"
RELS = META / "relations.csv"
STATUS = META / "doc_status.csv"
RAW = Path("data/raw_pdfs")
GATE = 5  # feasibility gate: at least this many docs must have a text-evidence-based relation


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


_cache = {}


def page_text(file, page):
    if file not in _cache:
        try:
            _cache[file] = load_pages(RAW / file)
        except Exception:
            _cache[file] = []
    pages = _cache[file]
    return pages[page - 1] if 1 <= page <= len(pages) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-quotes", action="store_true", help="skip quote-in-PDF check (fast)")
    args = ap.parse_args()

    if not SOURCES.exists():
        print("sources.csv not found (Step 1).")
        sys.exit(1)
    sources = {r["file"]: r for r in read_csv(SOURCES) if r.get("file")}

    if not RELS.exists():
        with RELS.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(REL_FIELDS)
        print(f"Created an empty template: {RELS}\nNow fill in the relations and run this script again.")
        return

    all_rows = read_csv(RELS)
    errors, warnings = validate(all_rows, sources, None if args.no_quotes else page_text)
    rels = [r for r in all_rows if any((v or "").strip() for v in r.values())]

    rc = Counter(r["relation"] for r in rels)
    print(f"relations.csv: {len(rels)} rows | " + " | ".join(f"{k} {v}" for k, v in sorted(rc.items())))
    ev = Counter(r["evidence_source"] for r in rels)
    fb = Counter(r["found_by"] for r in rels)
    print("evidence: " + " | ".join(f"{k} {v}" for k, v in sorted(ev.items())))
    print("found_by: " + " | ".join(f"{k} {v}" for k, v in sorted(fb.items())))

    text_docs = {r["src_file"] for r in rels
                 if r["evidence_source"] == "pdf_text"
                 and r["relation"] in ("supersedes", "amends", "withdraws")}
    print(f"docs with >=1 text-evidence relation: {len(text_docs)} / {len(sources)}  "
          f"(gate >= {GATE}: {'PASS' if len(text_docs) >= GATE else 'NOT YET'})")

    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\nerrors: {len(errors)} | warnings: {len(warnings)}")

    if errors:
        print("doc_status.csv was not updated (fix the errors first).")
        sys.exit(1)

    st = derive_status(rels, sources)
    fields = ["file", "status", "superseded_by", "amended_by", "withdrawn_by", "latest_active"]
    with STATUS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fn in sorted(st):
            w.writerow({"file": fn, **st[fn]})
    sc = Counter(v["status"] for v in st.values())
    print("status: " + " | ".join(f"{k} {v}" for k, v in sorted(sc.items())))
    print(f"doc_status.csv: {STATUS}")


if __name__ == "__main__":
    main()
