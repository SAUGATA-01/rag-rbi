"""Step 2: PyMuPDF vs pdfplumber comparison on data/raw_pdfs."""
import argparse
import csv
import re
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

RAW = Path("data/raw_pdfs")
DUMP = Path("data/processed/parse_compare")
RESULTS = Path("eval/results/step2_parser_compare.csv")

# numbered clause: "5.2.1 text" ya "12. text"  (sirf saal jaisa "2025 text" match nahi hoga)
CLAUSE = re.compile(r"^\s*(?:\d{1,3}(?:\.\d{1,3})+\.?|\d{1,3}\.)\s+\S")
# sub-clause: "(a) text", "(ii) text", "(3) text"
SUBCLAUSE = re.compile(r"^\s*\((?:[a-z]{1,3}|[ivx]{1,5}|\d{1,2})\)\s+\S")


def run_pymupdf(path, max_pages):
    texts, n_tables, t_text, t_tab = [], 0, 0.0, 0.0
    with fitz.open(path) as doc:
        for i in range(min(max_pages, len(doc))):
            page = doc[i]
            t = time.perf_counter()
            texts.append(page.get_text("text", sort=True))
            t_text += time.perf_counter() - t
            t = time.perf_counter()
            try:
                n_tables += len(page.find_tables().tables)
            except Exception:
                pass
            t_tab += time.perf_counter() - t
    return texts, n_tables, t_text, t_tab


def run_pdfplumber(path, max_pages):
    texts, n_tables, t_text, t_tab = [], 0, 0.0, 0.0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:max_pages]:
            t = time.perf_counter()
            texts.append(page.extract_text() or "")
            t_text += time.perf_counter() - t
            t = time.perf_counter()
            try:
                n_tables += len(page.find_tables())
            except Exception:
                pass
            t_tab += time.perf_counter() - t
    return texts, n_tables, t_text, t_tab


def stats(texts):
    lines_per_page = [[l.strip() for l in t.splitlines() if l.strip()] for t in texts]
    all_lines = [l for ls in lines_per_page for l in ls]
    n_pages = len(texts)
    seen = Counter()
    for ls in lines_per_page:
        for l in set(ls):
            if 5 <= len(l) <= 120:
                seen[l] += 1
    # header/footer proxy: line jo >= 50% pages par repeat hoti hai
    repeated = sum(1 for c in seen.values() if n_pages >= 4 and c >= n_pages * 0.5)
    short = sum(1 for l in all_lines if len(l.split()) <= 3)
    return {
        "chars": sum(len(t) for t in texts),
        "empty_pages": sum(1 for t in texts if len(t.strip()) < 20),
        "clause_lines": sum(1 for l in all_lines if CLAUSE.match(l)),
        "subclause_lines": sum(1 for l in all_lines if SUBCLAUSE.match(l)),
        "repeated_lines": repeated,
        "short_line_pct": round(100 * short / max(1, len(all_lines)), 1),
    }


def word_set(texts):
    return set(" ".join(texts).lower().split())


def overlap_pct(a, b):
    sa, sb = word_set(a), word_set(b)
    return round(100 * len(sa & sb) / max(1, len(sa | sb)), 1)


def dump(name, parser, texts):
    DUMP.mkdir(parents=True, exist_ok=True)
    pick = sorted({0, 1, 2, len(texts) // 2})
    with (DUMP / f"{Path(name).stem}__{parser}.txt").open("w", encoding="utf-8") as f:
        for i in pick:
            if i < len(texts):
                f.write(f"\n===== page {i + 1} =====\n{texts[i]}\n")


def selftest():
    sample = ["5.2.1 The bank shall comply.\n(a) first item\n(ii) second item\n"
              "2025 Directions apply.\n12. Definitions\n"]
    s = stats(sample)
    assert s["clause_lines"] == 2, s      # "5.2.1 ..." aur "12. ..."
    assert s["subclause_lines"] == 2, s   # "(a)" aur "(ii)"
    assert overlap_pct(["a b c"], ["b c d"]) == 50.0
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--only", default="", help="filename mein ye text ho to hi chalao")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    pdfs = [p for p in sorted(RAW.glob("*.pdf")) if args.only in p.name]
    if not pdfs:
        print("Koi PDF nahi mili.")
        return

    fields = ["file", "parser", "pages", "text_s", "tables_s", "tables", "chars",
              "empty_pages", "clause_lines", "subclause_lines", "repeated_lines",
              "short_line_pct", "word_overlap_pct", "error"]
    rows = []
    for p in pdfs:
        print(f"-> {p.name}", flush=True)
        out = {}
        for parser, fn in (("pymupdf", run_pymupdf), ("pdfplumber", run_pdfplumber)):
            row = {k: "" for k in fields}
            row.update(file=p.name, parser=parser)
            try:
                texts, n_tab, t_text, t_tab = fn(p, args.max_pages)
                row.update(pages=len(texts), text_s=round(t_text, 2),
                           tables_s=round(t_tab, 2), tables=n_tab, **stats(texts))
                dump(p.name, parser, texts)
                out[parser] = texts
            except Exception as e:
                row["error"] = str(e)[:150]
                out[parser] = []
            rows.append(row)
        ov = overlap_pct(out["pymupdf"], out["pdfplumber"])
        for r in rows[-2:]:
            r["word_overlap_pct"] = ov

    with RESULTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def tot(parser, key):
        return sum(float(r[key]) for r in rows if r["parser"] == parser and r[key] != "")

    def avg(parser, key):
        vals = [float(r[key]) for r in rows if r["parser"] == parser and r[key] != ""]
        return round(sum(vals) / max(1, len(vals)), 1)

    print("\nparser       text_s  tables_s  tables  clause  subcl  repeated  short%  empty")
    for pr in ("pymupdf", "pdfplumber"):
        print(f"{pr:<12} {tot(pr,'text_s'):>6.1f}  {tot(pr,'tables_s'):>8.1f}  "
              f"{tot(pr,'tables'):>6.0f}  {tot(pr,'clause_lines'):>6.0f}  "
              f"{tot(pr,'subclause_lines'):>5.0f}  {avg(pr,'repeated_lines'):>8}  "
              f"{avg(pr,'short_line_pct'):>6}  {tot(pr,'empty_pages'):>5.0f}")
    print(f"\navg word overlap: {avg('pymupdf', 'word_overlap_pct')}%")
    print(f"errors: {sum(1 for r in rows if r['error'])}")
    print(f"CSV: {RESULTS}\nDumps: {DUMP}")


if __name__ == "__main__":
    main()