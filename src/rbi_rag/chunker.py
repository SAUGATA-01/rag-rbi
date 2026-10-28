
"""Clause-aware chunker for RBI PDFs (Step 3). Pure Python, no ML libraries.

Pipeline: page texts -> clean (header/footer/page-no) -> segments (heading + numbered
clause boundaries) -> merge tiny segments -> split huge ones -> chunk records.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    min_words: int = 40          # segments shorter than this will be merged into the next segment
    max_words: int = 350         # segments longer than this will be split at sub-clause / sentence boundaries
    repeat_ratio: float = 0.5    # if a line appears on this percentage of pages, treat it as a header/footer
    min_pages_for_repeat: int = 4


# ---------------- regexes ----------------
# page number lines: "12", "Page 3 of 10", "- 4 -"
PAGE_NO = re.compile(
    r"^\s*(?:page\s+)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?\s*$|^\s*-\s*\d{1,3}\s*-\s*$", re.I
)
# numbered clause: "5.2.1 Text" or "12. Text" (dates such as "25 June 2025" are not treated as clauses)
CLAUSE_START = re.compile(r"^(?:(\d{1,3}(?:\.\d{1,3})+)\.?|(\d{1,3})\.)\s+(?=[A-Z(\[\"'\u201c\u2018])")
# sub-clause: "(a) text", "(ii) text", "(3) text"
SUBCLAUSE = re.compile(r"^\((?:[a-z]{1,3}|[ivx]{1,5}|\d{1,2})\)\s+\S")
HEADING_KW = re.compile(
    r"^(?:chapter|annex(?:ure)?|appendix|part|schedule)\s+(?:[ivxlc]+|\d{1,2}|[a-z])\b.{0,90}$", re.I
)


@dataclass
class Segment:
    heading: str
    clause_ids: list
    text: str
    page_start: int
    page_end: int
    n_lines: int = 0   # raw lines (for table-fragment detection)
    n_short: int = 0   # raw lines with <= 3 words


def wc(s: str) -> int:
    return len(s.split())


def is_heading(line: str) -> bool:
    if len(line) > 100 or line.endswith((".", ";", ",")):
        return False
    words = line.split()
    if HEADING_KW.match(line) and len(words) <= 12:
        return True
    if not any(c.isalpha() for c in line) or any(c.isdigit() for c in line):
        return False
    return line.isupper() and len(words) <= 12 and (len(words) >= 2 or len(line) >= 6)


# ---------------- 1. clean ----------------
def clean_pages(pages_text, cfg: ChunkConfig):
    pages = [[ln.strip() for ln in t.splitlines() if ln.strip()] for t in pages_text]
    n = len(pages)
    seen = Counter()
    for lines in pages:
        for ln in set(lines):
            if 5 <= len(ln) <= 120:
                seen[ln] += 1
    noise = set()
    if n >= cfg.min_pages_for_repeat:
        noise = {ln for ln, c in seen.items() if c >= n * cfg.repeat_ratio}
    return [[ln for ln in lines if ln not in noise and not PAGE_NO.match(ln)] for lines in pages]


# ---------------- 2. segments ----------------
def build_segments(pages_lines):
    segs, cur = [], None
    heading, heading_open = "Front matter", False

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            segs.append(Segment(cur["heading"], cur["ids"], cur["text"].strip(),
                                cur["p0"], cur["p1"], cur["n"], cur["short"]))
        cur = None

    for pno, lines in enumerate(pages_lines, start=1):
        for ln in lines:
            if is_heading(ln):
                if heading_open:                 # multi-line heading
                    heading = f"{heading} {ln}"
                else:
                    flush()
                    heading, heading_open = ln, True
                continue
            heading_open = False
            m = CLAUSE_START.match(ln)
            if m:
                flush()
                cur = {"heading": heading, "ids": [m.group(1) or m.group(2)], "text": ln,
                       "p0": pno, "p1": pno, "n": 1, "short": int(wc(ln) <= 3)}
                continue
            if cur is None:
                cur = {"heading": heading, "ids": [], "text": "", "p0": pno, "p1": pno,
                       "n": 0, "short": 0}
            sep = "\n" if SUBCLAUSE.match(ln) else " "
            cur["text"] = (cur["text"] + sep + ln) if cur["text"] else ln
            cur["p1"] = pno
            cur["n"] += 1
            cur["short"] += int(wc(ln) <= 3)
    flush()
    return segs


def _join(a: Segment, b: Segment) -> Segment:
    return Segment(b.heading, a.clause_ids + b.clause_ids, a.text + "\n" + b.text,
                   a.page_start, b.page_end, a.n_lines + b.n_lines, a.n_short + b.n_short)


# ---------------- 3. merge small ----------------
def merge_small(segs, cfg: ChunkConfig):
    out, carry = [], None
    for s in segs:
        if carry is not None:
            if carry.heading == s.heading:       # do not merge across headings
                s = _join(carry, s)
            else:
                out.append(carry)
            carry = None
        if wc(s.text) < cfg.min_words:
            carry = s
        else:
            out.append(s)
    if carry is not None:
        if out and out[-1].heading == carry.heading and wc(out[-1].text) + wc(carry.text) <= cfg.max_words:
            out[-1] = _join(out[-1], carry)
        else:
            out.append(carry)
    return out


# ---------------- 4. split big ----------------
def pack(units, max_words):
    parts, cur, n = [], [], 0
    for u in units:
        w = wc(u)
        if cur and n + w > max_words:
            parts.append("\n".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += w
    if cur:
        parts.append("\n".join(cur))
    return parts


def split_big(seg: Segment, cfg: ChunkConfig):
    if wc(seg.text) <= cfg.max_words:
        return [seg]
    units = []
    for line in seg.text.split("\n"):
        if wc(line) > cfg.max_words:             # if the sub-clause is also too long, split at sentence boundaries
            units += [u for u in re.split(r"(?<=[.;:])\s+", line) if u]
        else:
            units.append(line)
    parts = pack(units, cfg.max_words)
    if len(parts) > 1 and wc(parts[-1]) < cfg.min_words // 2:   # do not leave a short tail
        parts[-2] = parts[-2] + "\n" + parts.pop()
    return [Segment(seg.heading, seg.clause_ids, p, seg.page_start, seg.page_end,
                    seg.n_lines, seg.n_short) for p in parts]


# ---------------- 5. chunk records ----------------
def to_chunks(segs, stem, meta):
    chunks = []
    for i, s in enumerate(segs):
        clause = ", ".join(s.clause_ids)
        ctx = " | ".join(x for x in [meta.get("title", ""), s.heading,
                                     f"Clause {clause}" if clause else ""] if x)
        chunks.append({
            "chunk_id": f"{stem}::{i:04d}",
            "file": meta.get("file", stem + ".pdf"),
            "title": meta.get("title", ""),
            "circular_no": meta.get("circular_no", ""),
            "date": meta.get("date", ""),
            "doc_type": meta.get("type", ""),
            "heading": s.heading,
            "clause_ids": clause,
            "page_start": s.page_start,
            "page_end": s.page_end,
            "n_words": wc(s.text),
            "maybe_table": s.n_lines >= 8 and s.n_short / s.n_lines >= 0.6,
            "text": s.text,
            "embed_text": f"{ctx}\n{s.text}" if ctx else s.text,
        })
    return chunks


def chunk_pages(pages_text, stem, meta=None, cfg: ChunkConfig | None = None):
    cfg = cfg or ChunkConfig()
    meta = meta or {}
    segs = merge_small(build_segments(clean_pages(pages_text, cfg)), cfg)
    final = [p for s in segs for p in split_big(s, cfg)]
    return to_chunks(final, stem, meta)


def load_pages(pdf_path, parser="pymupdf", max_pages=None):
    if parser == "pymupdf":
        import fitz
        with fitz.open(pdf_path) as doc:
            n = len(doc) if max_pages is None else min(max_pages, len(doc))
            return [doc[i].get_text("text", sort=True) for i in range(n)]
    if parser == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            return [p.extract_text() or "" for p in pages]
    raise ValueError(f"unknown parser: {parser}")
