
"""Step 4: Ground-truth relationships (amends / supersedes / withdraws) between RBI documents.

relations.csv (manually populated) -> validate -> derive doc_status.csv.
The candidate finder is only a labeling aid, not an extraction tool.
"""
from __future__ import annotations

import re
import unicodedata

REL_FIELDS = ["src_file", "relation", "dst_file", "dst_ref", "evidence_source",
              "evidence_page", "evidence_quote", "found_by", "notes"]
VALID_REL = {"supersedes", "amends", "withdraws", "refers_to"}
VALID_EVID = {"pdf_text", "rbi_web"}
VALID_FOUND = {"manual", "finder"}



def norm(s: str) -> str:
    """Normalize hyphenated line breaks, Unicode text, punctuation, and whitespace."""
    s = re.sub(r"-\s*\n\s*", "", s or "")
    s = unicodedata.normalize("NFC", s).casefold()
    s = "".join(
        ch for ch in s
        if unicodedata.category(ch)[0] != "P"
    )
    return re.sub(r"\s+", " ", s).strip()




# ---------------- validation ----------------
def _find_cycle(edges):
    """edges: dict mapping nodes to lists of nodes. Returns a list of cycle nodes if found, otherwise None."""
    state, stack = {}, []

    def dfs(n):
        state[n] = 1
        stack.append(n)
        for m in edges.get(n, []):
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if state.get(m) is None:
                c = dfs(m)
                if c:
                    return c
        stack.pop()
        state[n] = 2
        return None

    for n in list(edges):
        if state.get(n) is None:
            c = dfs(n)
            if c:
                return c
    return None


def validate(rels, sources, page_text=None):
    """rels: list[dict]; sources: dict mapping file names to rows; page_text(file, page) -> str | None.
    Returns (errors, warnings), with each entry formatted as 'row N: ...'."""
    errors, warnings = [], []
    seen = set()
    sup_edges = {}
    for i, r in enumerate(rels, start=2):  # Row 1 contains the CSV header
        tag = f"row {i}"
        if not any((v or "").strip() for v in r.values()):
            continue  # Ignore empty spreadsheet rows
        src, rel, dst = r.get("src_file", ""), r.get("relation", ""), r.get("dst_file", "")
        if src not in sources:
            errors.append(f"{tag}: src_file '{src}' does not exist in sources.csv")
            continue
        if rel not in VALID_REL:
            errors.append(f"{tag}: invalid relation '{rel}' (allowed: {sorted(VALID_REL)})")
        if dst and dst not in sources:
            errors.append(f"{tag}: dst_file '{dst}' does not exist in sources.csv (leave blank if the target is outside the corpus)")
        if dst and dst == src:
            errors.append(f"{tag}: source and destination are the same")
        if not r.get("dst_ref", "").strip():
            errors.append(f"{tag}: dst_ref is empty (enter the target's name as cited in the document)")
        ev = r.get("evidence_source", "")
        if ev not in VALID_EVID:
            errors.append(f"{tag}: invalid evidence_source '{ev}' (expected pdf_text / rbi_web)")
        if r.get("found_by", "") not in VALID_FOUND:
            errors.append(f"{tag}: invalid found_by '{r.get('found_by', '')}' (expected manual / finder)")

        quote = (r.get("evidence_quote") or "").strip()
        if ev == "pdf_text":
            page_s = (r.get("evidence_page") or "").strip()
            if not page_s.isdigit() or int(page_s) < 1:
                errors.append(f"{tag}: evidence_page must be a valid PDF viewer page number")
            elif not quote:
                errors.append(f"{tag}: evidence_quote is empty")
            else:
                if len(quote.split()) > 40:
                    warnings.append(f"{tag}: quote exceeds 40 words; shorten it")
                if page_text is not None:
                    page = int(page_s)
                    nq = norm(quote)
                    here = page_text(src, page)
                    if here is None:
                        errors.append(f"{tag}: page {page} does not exist in {src}")
                    elif nq not in norm(here):
                        near = [p for p in (page - 1, page + 1)
                                if p >= 1 and (page_text(src, p) is not None)
                                and nq in norm(page_text(src, p) or "")]
                        if near:
                            errors.append(f"{tag}: quote not found on page {page}; found on page {near[0]} instead")
                        else:
                            errors.append(f"{tag}: quote not found on page {page} of {src} or on adjacent pages")
        elif ev == "rbi_web" and not (r.get("notes") or "").strip():
            warnings.append(f"{tag}: evidence is from the RBI website, but notes do not contain a URL")

        key = (src, rel, dst or r.get("dst_ref", "").strip().lower())
        if key in seen:
            warnings.append(f"{tag}: duplicate relation {key}")
        seen.add(key)

        if rel in ("supersedes", "withdraws") and dst:
            sup_edges.setdefault(src, []).append(dst)
            d_src = (sources[src].get("date") or "").strip()
            d_dst = (sources.get(dst, {}).get("date") or "").strip()
            if d_src and d_dst and d_src < d_dst:
                warnings.append(f"{tag}: {src} ({d_src}) is older than {dst} ({d_dst}), but is labeled '{rel}'")
    cyc = _find_cycle(sup_edges)
    if cyc:
        errors.append("cycle in supersedes/withdraws: " + " -> ".join(cyc))
    return errors, warnings


# ---------------- status derivation ----------------
def derive_status(rels, sources):
    sup_by, amd_by, wd_by = {}, {}, {}
    for r in rels:
        dst, src, rel = r.get("dst_file", ""), r.get("src_file", ""), r.get("relation", "")
        if not dst or dst not in sources or src not in sources:
            continue
        {"supersedes": sup_by, "amends": amd_by, "withdraws": wd_by}.get(rel, {}).setdefault(dst, []).append(src)

    def walk(f, seen):
        if f in seen:
            return []
        seen.add(f)
        nxt = sup_by.get(f, [])
        if not nxt:
            return [] if f in wd_by else [f]
        out = []
        for n in nxt:
            out += walk(n, seen)
        return out

    out = {}
    for f in sources:
        if sup_by.get(f):
            status = "superseded"
        elif wd_by.get(f):
            status = "withdrawn"
        else:
            status = "active"
        latest = [] if status == "withdrawn" else walk(f, set())
        out[f] = {
            "status": status,
            "superseded_by": ";".join(sup_by.get(f, [])),
            "amended_by": ";".join(amd_by.get(f, [])),
            "withdrawn_by": ";".join(wd_by.get(f, [])),
            "latest_active": ";".join(dict.fromkeys(latest)),
        }
    return out


# ---------------- candidate finder (labeling helper) ----------------
TRIGGERS = {
    "supersede": re.compile(r"\bsuperse(?:de|ded|des|ding)\b|\bin supersession\b", re.I),
    "repeal": re.compile(r"\brepeal(?:ed|s|ing)?\b", re.I),
    "rescind": re.compile(r"\brescind(?:ed|s)?\b", re.I),
    "withdraw": re.compile(r"\bwithdraw(?:n|s|al)?\b", re.I),
    "amend": re.compile(r"\bamend(?:ed|s|ing|ment|ments)?\b", re.I),
    "modify": re.compile(r"\bmodif(?:ied|ies|y|ication|ications)\b", re.I),
    "replace": re.compile(r"\b(?:replaced|replaces|replacing)\b", re.I),
    "read_with": re.compile(r"\bread with\b", re.I),
}
_SENT = re.compile(r"(?<=[.;])\s+")


def find_candidates(chunks):
    out = []
    for c in chunks:
        for sent in _SENT.split(c["text"].replace("\n", " ")):
            hits = [n for n, rx in TRIGGERS.items() if rx.search(sent)]
            if hits:
                out.append({"file": c["file"], "chunk_id": c["chunk_id"], "page": c["page_start"],
                            "triggers": "|".join(hits), "sentence": sent.strip()[:400]})
    return out
