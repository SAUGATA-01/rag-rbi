"""Step 6: golden dataset - schema, validation, plan, helpers (probe, version diff, drafts)."""
from __future__ import annotations

import difflib
import json
import math
import random
import re
from collections import Counter, defaultdict

from rbi_rag.relations import norm

CATEGORIES = ("simple", "multihop", "outdated_trap", "what_changed", "unanswerable")
STATUSES = ("draft", "verified", "rejected")
SPLITS = ("dev", "test")
AUTHORS = ("self", "llm_assisted")
UNANS_SUBTYPES = ("out_of_scope", "near_miss", "false_premise")
UNANSWERABLE = "UNANSWERABLE"
REL_KINDS = ("supersedes", "amends", "withdraws")

FIELDS = ["id", "split", "category", "graph_only", "subtype", "question", "reference_answer",
          "e1_file", "e1_page", "e1_quote", "e2_file", "e2_page", "e2_quote",
          "e3_file", "e3_page", "e3_quote", "trap_file", "trap_page", "trap_quote",
          "authored_by", "status", "notes"]

TARGET_DEV = {"simple": 15, "multihop": 12, "outdated_trap": 12, "what_changed": 9, "unanswerable": 12}
GRAPH_ONLY_DEV = 6      # multihop mein se itne sirf graph se solve hone chahiye
MIN_SELF = 12           # khud likhe hue (LLM ke bina) kam se kam
OVERLAP_WARN = 0.7

STOP = set("a an the of to in on for and or is are was were be been by with as at from that this it its "
           "under shall may can what which who how when where does do did if not any all such than then "
           "there their these those into per".split())


# ---------------- small helpers ----------------
def content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


def overlap_coef(question: str, quote: str) -> float:
    q, e = content_words(question), content_words(quote)
    return len(q & e) / max(1, len(q))


def evidence(r: dict) -> list:
    out = []
    for i in (1, 2, 3):
        f, p, q = (r.get(f"e{i}_file") or "").strip(), (r.get(f"e{i}_page") or "").strip(), (r.get(f"e{i}_quote") or "").strip()
        if f or p or q:
            out.append((f, p, q))
    return out


def check_quote(file, page_s, quote, page_text):
    """None = ok, warna error string."""
    if not page_s.isdigit() or int(page_s) < 1:
        return "page number nahi hai"
    if not quote:
        return "quote khaali hai"
    if page_text is None:
        return None
    page, nq = int(page_s), norm(quote)
    here = page_text(file, page)
    if here is None:
        return f"{file} mein page {page} nahi hai"
    if nq in norm(here):
        return None
    near = [p for p in (page - 1, page + 1) if p >= 1 and nq in norm(page_text(file, p) or "")]
    if near:
        return f"quote page {page} par nahi, page {near[0]} par mila"
    return f"quote {file} ke page {page} (aur aas-paas) mein nahi mila"


def has_rel(rels, src, dst) -> bool:
    return any(r.get("relation") in REL_KINDS and r.get("src_file") == src and r.get("dst_file") == dst
               for r in rels)


def connected(rels, a, b) -> bool:
    return has_rel(rels, a, b) or has_rel(rels, b, a)


# ---------------- validation ----------------
def validate(rows, sources, status, rels, page_text=None):
    errors, warns, seen = [], [], set()
    for i, r in enumerate(rows, start=2):
        if not any((v or "").strip() for v in r.values()):
            continue
        qid = (r.get("id") or "").strip()
        tag = f"row {i} ({qid or '?'})"

        def err(m, tag=tag):
            errors.append(f"{tag}: {m}")

        if not qid:
            err("id khaali")
            continue
        if qid in seen:
            err("duplicate id")
        seen.add(qid)
        st = (r.get("status") or "").strip()
        if st not in STATUSES:
            err(f"status '{st}' invalid {STATUSES}")
            continue
        if st != "verified":
            continue  # draft / rejected par deep check nahi

        cat = (r.get("category") or "").strip()
        if cat not in CATEGORIES:
            err(f"category '{cat}' invalid {CATEGORIES}")
            continue
        if (r.get("split") or "").strip() not in SPLITS:
            err("split dev/test hona chahiye")
        if (r.get("authored_by") or "").strip() not in AUTHORS:
            err("authored_by self/llm_assisted hona chahiye")
        q = (r.get("question") or "").strip()
        if not q:
            err("question khaali")
        elif len(q.split()) < 6:
            warns.append(f"{tag}: question bahut chhota hai")
        go = (r.get("graph_only") or "").strip().lower() == "yes"
        if go and cat != "multihop":
            err("graph_only sirf multihop category mein")

        ans = (r.get("reference_answer") or "").strip()
        ev = evidence(r)

        if cat == "unanswerable":
            if ans != UNANSWERABLE:
                err(f"reference_answer '{UNANSWERABLE}' hona chahiye")
            if (r.get("subtype") or "").strip() not in UNANS_SUBTYPES:
                err(f"subtype {UNANS_SUBTYPES} mein se ho")
            if not (r.get("notes") or "").strip():
                err("notes mein likho ki ye unanswerable kyu hai (probe se confirm kiya?)")
            if ev:
                warns.append(f"{tag}: unanswerable mein evidence nahi hona chahiye")
            continue

        if not ans:
            err("reference_answer khaali")
        if not ev:
            err("kam se kam e1 evidence chahiye")
        files = []
        for n, (f, p, qt) in enumerate(ev, start=1):
            if f not in sources:
                err(f"e{n}_file '{f}' sources.csv mein nahi hai")
                continue
            files.append(f)
            msg = check_quote(f, p, qt, page_text)
            if msg:
                err(f"e{n}: {msg}")
            elif qt and overlap_coef(q, qt) > OVERLAP_WARN:
                warns.append(f"{tag}: question e{n} quote se bahut milta hai (paraphrase karo)")
        if not files:
            continue

        def st_of(f):
            return (status.get(f) or {}).get("status", "unknown")

        if cat == "simple":
            if st_of(files[0]) != "active":
                err(f"simple ka evidence doc active hona chahiye ({files[0]} = {st_of(files[0])})")
        elif cat == "multihop":
            if len(set(files)) < 2:
                err("multihop mein kam se kam 2 alag documents ka evidence chahiye")
            elif go:
                if not any(connected(rels, a, b) for a in set(files) for b in set(files) if a < b):
                    err("graph_only hai par evidence docs ke beech relations.csv mein koi relation nahi")
            else:
                bad = [f for f in set(files) if st_of(f) != "active"]
                if bad:
                    warns.append(f"{tag}: non-graph multihop mein inactive doc: {bad}")
        elif cat == "what_changed":
            if len(set(files)) < 2:
                err("what_changed mein e1 = naya/amending doc, e2 = purana/base doc chahiye")
            elif not has_rel(rels, files[0], files[1]):
                err(f"relations.csv mein '{files[0]} -> {files[1]}' (supersedes/amends/withdraws) nahi hai")
        elif cat == "outdated_trap":
            tf, tp, tq = ((r.get("trap_file") or "").strip(), (r.get("trap_page") or "").strip(),
                          (r.get("trap_quote") or "").strip())
            if tf not in sources:
                err(f"trap_file '{tf}' sources.csv mein nahi hai")
            else:
                if st_of(tf) != "superseded":
                    err(f"trap_file superseded hona chahiye ({tf} = {st_of(tf)})")
                latest = ((status.get(tf) or {}).get("latest_active") or "").split(";")
                if files[0] not in latest:
                    err(f"e1_file {files[0]} trap_file ka latest_active nahi hai ({latest})")
                if tf == files[0]:
                    err("trap_file aur e1_file same nahi ho sakte")
                msg = check_quote(tf, tp, tq, page_text)
                if msg:
                    err(f"trap: {msg}")
    return errors, warns


def summarize(rows):
    live = [r for r in rows if (r.get("status") or "").strip() == "verified"]
    return {
        "verified": len(live),
        "draft": sum(1 for r in rows if (r.get("status") or "").strip() == "draft"),
        "rejected": sum(1 for r in rows if (r.get("status") or "").strip() == "rejected"),
        "by_category": dict(Counter(r["category"] for r in live)),
        "graph_only": sum(1 for r in live if (r.get("graph_only") or "").strip().lower() == "yes"),
        "self_authored": sum(1 for r in live if (r.get("authored_by") or "").strip() == "self"),
        "by_split": dict(Counter(r["split"] for r in live)),
    }


def doc_share(rows):
    """Answerable verified questions mein har doc ka hissa (ek doc ka dabdaba na ho)."""
    c = Counter()
    n = 0
    for r in rows:
        if (r.get("status") or "").strip() != "verified" or r.get("category") == "unanswerable":
            continue
        n += 1
        for f in {e[0] for e in evidence(r)}:
            c[f] += 1
    return {f: round(100 * k / max(1, n), 1) for f, k in c.most_common()}


def to_record(r):
    ev = [{"file": f, "page": int(p), "quote": q} for f, p, q in evidence(r)]
    trap = None
    if (r.get("trap_file") or "").strip():
        trap = {"file": r["trap_file"].strip(), "page": int(r["trap_page"]), "quote": r["trap_quote"].strip()}
    return {"id": r["id"].strip(), "split": r["split"].strip(), "category": r["category"].strip(),
            "graph_only": (r.get("graph_only") or "").strip().lower() == "yes",
            "subtype": (r.get("subtype") or "").strip(), "question": r["question"].strip(),
            "reference_answer": r["reference_answer"].strip(), "evidence": ev, "trap": trap,
            "expected_docs": sorted({e["file"] for e in ev}), "authored_by": r["authored_by"].strip()}


# ---------------- plan ----------------
def build_plan(chunks, status, rels, seed=7, targets=None):
    targets = targets or TARGET_DEV
    rnd = random.Random(seed)
    rows, warns = [], []
    n = [0]

    def add(cat, hint, files="", page="", graph="", subtype="", packet="no"):
        n[0] += 1
        rows.append({"slot_id": f"P{n[0]:02d}", "category": cat, "graph_only": graph, "subtype": subtype,
                     "target_files": files, "target_page": page, "packet": packet, "hint": hint})

    # simple: sirf active docs, round-robin, random page
    by_doc = defaultdict(list)
    for c in chunks:
        if (status.get(c["file"]) or {}).get("status") == "active" and c["n_words"] >= 60 and not c.get("maybe_table"):
            by_doc[c["file"]].append(c)
    docs = sorted(by_doc)
    rnd.shuffle(docs)
    if len(docs) < targets["simple"]:
        warns.append(f"simple ke liye sirf {len(docs)} active docs mile (docs repeat honge)")
    for k in range(targets["simple"] if docs else 0):
        d = docs[k % len(docs)]
        c = rnd.choice(by_doc[d])
        add("simple", f"{d} page {c['page_start']} se ek tathya-wala sawal", d, c["page_start"], packet="yes")

    edges = [(r["src_file"], r["relation"], r["dst_file"]) for r in rels
             if r.get("dst_file") and r.get("relation") in REL_KINDS]
    sup = [e for e in edges if e[1] == "supersedes"]
    amd = [e for e in edges if e[1] == "amends"]

    def cycle(lst, k):
        return [lst[i % len(lst)] for i in range(k)] if lst else []

    if len(sup) < 4:
        warns.append(f"supersedes pairs sirf {len(sup)} hain (outdated_trap ke liye 4+ chahiye)")
    for s, _, d in cycle(sup, targets["outdated_trap"]):
        add("outdated_trap", f"naya={s}, purana={d}: aisa provision dhoondo jo badla ho (diff subcommand). "
                             f"Slot repeat ho to alag provision lo", f"{s};{d}")
    wc = amd + sup
    if len(amd) < 3:
        warns.append(f"amends pairs sirf {len(amd)} hain (what_changed ke liye 3+ chahiye)")
    for s, rel, d in cycle(wc, targets["what_changed"]):
        add("what_changed", f"{s} ne {d} mein kya badla ({rel})", f"{s};{d}")

    cv = [f for f, s in status.items() if s.get("superseded_by")]
    tr = [f for f, s in status.items() if s.get("amended_by")]
    stt = [f for f, s in status.items() if s.get("status") in ("superseded", "withdrawn")]
    kinds = [("current_version", cv, "latest_active"), ("amendment_trail", tr, "amended_by"),
             ("status", stt, "status")]
    kinds = [k for k in kinds if k[1]]
    for i in range(GRAPH_ONLY_DEV if kinds else 0):
        kind, lst, key = kinds[i % len(kinds)]
        f = lst[(i // len(kinds)) % len(lst)]
        add("multihop", f"graph-only [{kind}] doc={f}; sahi jawab = {status[f].get(key)}", f, graph="yes")
    if not kinds:
        warns.append("graph-only slots nahi bane (doc_status.csv mein relations nahi mili)")
    for _ in range(targets["multihop"] - GRAPH_ONLY_DEV):
        add("multihop", "do alag active documents jo ek hi topic par hon (probe use karo), "
                        "jawab dono ko jod ke bane", graph="no")

    for sub, k, hint in (("out_of_scope", 4, "RBI se bahar ka topic (jaise stock market, GST, kisi aur desh ka niyam)"),
                         ("near_miss", 5, "RBI ka topic jo tumhare corpus mein NAHI hai (probe se confirm karo)"),
                         ("false_premise", 3, "maujood document ke ek aise clause/paragraph ke baare mein poochho jo exist nahi karta")):
        for _ in range(k):
            add("unanswerable", hint, subtype=sub)
    return rows, warns


# ---------------- LLM-assisted drafting (packets / import) ----------------
PROMPT = """You are helping build an evaluation set for a question-answering system over RBI (Reserve Bank of India) documents.
Using ONLY the page text below, write 2 question-answer pairs that a banking student or compliance learner might ask.

Rules:
- The question must be self-contained (never say "this circular", "the above", "this page").
- Do NOT copy sentences from the page: paraphrase, use everyday wording.
- Each question needs ONE checkable answer found on this page. Answer in at most 2 sentences.
- Add a verbatim quote (8 to 25 words, from a single line of the page) that supports the answer.
- Output ONLY JSON lines, one object per line, no commentary:
{{"slot": "{slot}", "question": "...", "reference_answer": "...", "quote": "..."}}

Document: {title} ({circular_no}, {date})
Page: {page}

--- PAGE TEXT ---
{text}
"""


def parse_drafts(text: str):
    """LLM output (JSON lines / JSON array / markdown fence) -> (items, bad_line_count)."""
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    items, bad = [], 0
    try:
        obj = json.loads(t)
        objs = obj if isinstance(obj, list) else [obj]
        items = [o for o in objs if isinstance(o, dict)]
    except Exception:
        for line in t.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    items.append(o)
                else:
                    bad += 1
            except Exception:
                bad += 1
    good = [o for o in items if all((o.get(k) or "").strip() for k in ("slot", "question", "reference_answer", "quote"))]
    return good, bad + (len(items) - len(good))


# ---------------- probe (throwaway lexical helper, BM25 nahi) ----------------
def build_probe(chunks):
    toks = [content_words(c["text"]) for c in chunks]
    df = Counter()
    for t in toks:
        df.update(t)
    n = len(chunks)
    idf = {w: math.log((n + 1) / (d + 0.5)) for w, d in df.items()}

    def search(q, k=5):
        qs = content_words(q)
        scored = sorted(((sum(idf.get(w, 0.0) for w in qs & t), i) for i, t in enumerate(toks)), reverse=True)[:k]
        return [(round(s, 2), chunks[i]) for s, i in scored if s > 0]

    return search


# ---------------- version diff (outdated_trap / what_changed ke liye) ----------------
_NUM = re.compile(r"\d+(?:[.,]\d+)?\s?%?")


def numbers(s: str) -> set:
    return {m.group(0).replace(" ", "") for m in _NUM.finditer(s or "")}


def diff_pairs(old_chunks, new_chunks, min_ratio=0.5, top=15):
    old_t = [(norm(c["text"]).split(), content_words(c["text"])) for c in old_chunks]
    out = []
    for nc in new_chunks:
        nt, ns = norm(nc["text"]).split(), content_words(nc["text"])
        best, best_i = 0.0, -1
        for i, (ot, os_) in enumerate(old_t):
            if not (ns | os_) or len(ns & os_) / len(ns | os_) < 0.25:
                continue
            r = difflib.SequenceMatcher(None, ot, nt, autojunk=False).ratio()
            if r > best:
                best, best_i = r, i
        if best_i < 0 or best < min_ratio or best >= 0.995:
            continue
        oc = old_chunks[best_i]
        on, nn = numbers(oc["text"]), numbers(nc["text"])
        out.append({"ratio": round(best, 2), "old": oc, "new": nc,
                    "only_old": sorted(on - nn), "only_new": sorted(nn - on)})
    out.sort(key=lambda d: (-(len(d["only_old"]) + len(d["only_new"]) > 0), d["ratio"]))
    return out[:top]