"""Step 1: data/raw_pdfs scan -> manifest.csv (technical) + sources.csv (manual metadata)."""
import csv
from pathlib import Path

import fitz  # PyMuPDF

RAW = Path("data/raw_pdfs")
META = Path("data/metadata")
MANIFEST = META / "manifest.csv"
SOURCES = META / "sources.csv"
OLD_DOCS = META / "documents.csv"  # pichle Step 1 version ki file (agar bani ho)

MANIFEST_FIELDS = ["file", "size_kb", "pages", "chars_first3", "text_ok", "error"]
SOURCE_FIELDS = ["file", "title", "rbi_url", "year", "type", "circular_no", "date"]
REQUIRED = ["title", "rbi_url", "year", "type", "circular_no", "date"]
# purani documents.csv ke columns -> sources.csv ke columns
OLD_MAP = {"title": "title", "source_url": "rbi_url", "doc_type": "type",
           "circular_no": "circular_no", "issue_date": "date"}


def read_csv(path, key):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:  # Excel BOM safe
        return {r[key]: r for r in csv.DictReader(f) if r.get(key)}


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def inspect(pdf):
    row = {"file": pdf.name, "size_kb": round(pdf.stat().st_size / 1024),
           "pages": -1, "chars_first3": -1, "text_ok": False, "error": ""}
    try:
        with fitz.open(pdf) as doc:
            row["pages"] = len(doc)
            n = sum(len(doc[i].get_text().strip()) for i in range(min(3, len(doc))))
            row["chars_first3"] = n
            row["text_ok"] = n > 500
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


def guess_from_name(name, row):
    parts = Path(name).stem.split("_")
    if not row["type"] and parts[0].upper() in {"MD", "AMD", "CIR"}:
        row["type"] = parts[0].upper()
    if not row["year"] and parts[-1].isdigit() and len(parts[-1]) == 4:
        row["year"] = parts[-1]


def main():
    META.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        print(f"{RAW} mein koi PDF nahi mili. Pehle PDFs download karo.")
        return

    manifest = [inspect(p) for p in pdfs]
    write_csv(MANIFEST, MANIFEST_FIELDS, manifest)

    old = read_csv(OLD_DOCS, "filename")
    existing = read_csv(SOURCES, "file")
    rows = []
    for p in pdfs:
        row = {k: "" for k in SOURCE_FIELDS}
        row["file"] = p.name
        if p.name in old:  # migrate
            for o, n in OLD_MAP.items():
                row[n] = old[p.name].get(o, "") or ""
        for k, v in existing.get(p.name, {}).items():
            if k in SOURCE_FIELDS and v:
                row[k] = v
        guess_from_name(p.name, row)
        rows.append(row)

    names = {p.name for p in pdfs}
    orphans = [f for f in existing if f not in names]
    for f in orphans:  # manual kaam mat khona
        rows.append({k: existing[f].get(k, "") for k in SOURCE_FIELDS})
    write_csv(SOURCES, SOURCE_FIELDS, rows)

    ok = sum(1 for r in manifest if r["text_ok"])
    bad = [r["file"] for r in manifest if not r["text_ok"]]
    incomplete = [r["file"] for r in rows if any(not r[k] for k in REQUIRED)]

    print(f"PDFs: {len(manifest)} | text_ok: {ok} | problem: {len(bad)}")
    print(f"sources.csv: {len(rows)} rows | complete: {len(rows) - len(incomplete)} | incomplete: {len(incomplete)}")
    for b in bad:
        print("  [text problem]", b)
    for o in orphans:
        print("  [sources.csv mein hai par PDF nahi mili]", o)
    print(f"Files: {MANIFEST} , {SOURCES}")


if __name__ == "__main__":
    main()