"""Step 6: golden dataset tooling.

  plan      slots ki list banao (kaunsi category ke liye kaunse docs/pages)
  packets   simple slots ke liye LLM-drafting packets (raw PDF page text + prompt)
  import    LLM ke drafts (drafts.jsonl) ko questions.csv mein 'draft' ki tarah jodo
  probe     lexical helper: sawal ke liye corpus mein sabse milte chunks (unanswerable / multihop check)
  diff      do versions ke beech badle hue provisions (outdated_trap / what_changed ke liye)
  validate  verified questions check karo, golden.jsonl export karo
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from rbi_rag.chunker import load_pages
from rbi_rag.golden import (FIELDS, GRAPH_ONLY_DEV, MIN_SELF, PROMPT, TARGET_DEV, UNANSWERABLE,
                            build_plan, build_probe, diff_pairs, doc_share, parse_drafts,
                            summarize, to_record, validate)

RAW = Path("data/raw_pdfs")
META = Path("data/metadata")
CORPUS = Path("data/processed/corpus.jsonl")
PACKETS = Path("data/processed/golden_packets")
GOLD = Path("eval/golden")
PLAN, QUESTIONS, DRAFTS = GOLD / "plan.csv", GOLD / "questions.csv", GOLD / "drafts.jsonl"
EXPORT, GMAN = GOLD / "golden.jsonl", GOLD / "golden_manifest.json"
PLAN_FIELDS = ["slot_id", "category", "graph_only", "subtype", "target_files", "target_page", "packet", "hint"]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_corpus():
    if not CORPUS.exists():
        print("corpus.jsonl nahi mili. Pehle Step 5 chalao.")
        sys.exit(1)
    with CORPUS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_pages = {}


def page_text(file, page):
    if file not in _pages:
        try:
            _pages[file] = load_pages(RAW / file)
        except Exception:
            _pages[file] = []
    pages = _pages[file]
    return pages[page - 1] if 1 <= page <= len(pages) else None


def resolve(chunks, frag):
    files = sorted({c["file"] for c in chunks if frag in c["file"]})
    if len(files) != 1:
        print(f"'{frag}' se {len(files)} files mili {files[:5]}. Poora/unique naam do.")
        sys.exit(1)
    return files[0]


# ---------------- subcommands ----------------
def cmd_plan(a):
    if PLAN.exists() and not a.force:
        print(f"{PLAN} pehle se hai (tumhare notes ho sakte hain). Overwrite ke liye --force.")
        return
    chunks = load_corpus()
    status = {r["file"]: r for r in read_csv(META / "doc_status.csv")}
    rows, warns = build_plan(chunks, status, read_csv(META / "relations.csv"), seed=a.seed)
    write_csv(PLAN, PLAN_FIELDS, rows)
    by = {}
    for r in rows:
        by[r["category"]] = by.get(r["category"], 0) + 1
    print(f"plan: {len(rows)} slots -> {PLAN}")
    print(" | ".join(f"{k} {v}/{TARGET_DEV[k]}" for k, v in by.items()))
    for w in warns:
        print("WARN ", w)


def cmd_packets(a):
    plan = [r for r in read_csv(PLAN) if r["packet"] == "yes"]
    if not plan:
        print("plan.csv mein packet=yes wale slots nahi hain (pehle plan chalao).")
        return
    src = {r["file"]: r for r in read_csv(META / "sources.csv")}
    PACKETS.mkdir(parents=True, exist_ok=True)
    for r in plan:
        f, p = r["target_files"], int(r["target_page"])
        text = page_text(f, p)
        if text is None:
            print(f"SKIP {r['slot_id']}: {f} page {p} nahi mila")
            continue
        m = src.get(f, {})
        body = PROMPT.format(slot=r["slot_id"], title=m.get("title", ""), circular_no=m.get("circular_no", ""),
                             date=m.get("date", ""), page=p, text=text.strip())
        (PACKETS / f"{r['slot_id']}.txt").write_text(body, encoding="utf-8")
    print(f"{len(plan)} packets -> {PACKETS}\nHar packet ko free chat LLM mein paste karo, "
          f"JSON lines output {DRAFTS} mein append karo, phir `import` chalao.")


def cmd_import(a):
    if not DRAFTS.exists():
        print(f"{DRAFTS} nahi mili.")
        return
    good, bad = parse_drafts(DRAFTS.read_text(encoding="utf-8"))
    plan = {r["slot_id"]: r for r in read_csv(PLAN)}
    rows = read_csv(QUESTIONS)
    seen_q = {(r.get("question") or "").strip().lower() for r in rows}
    n = max([int(r["id"][1:]) for r in rows if (r.get("id") or "q")[1:].isdigit()] or [0])
    added, skipped = [], 0
    for d in good:
        slot = plan.get(d["slot"].strip())
        if not slot or d["question"].strip().lower() in seen_q:
            skipped += 1
            continue
        n += 1
        seen_q.add(d["question"].strip().lower())
        row = {k: "" for k in FIELDS}
        row.update(id=f"q{n:03d}", split="dev", category="simple", question=d["question"].strip(),
                   reference_answer=d["reference_answer"].strip(), e1_file=slot["target_files"],
                   e1_page=slot["target_page"], e1_quote=d["quote"].strip(), authored_by="llm_assisted",
                   status="draft", notes=f"slot={slot['slot_id']}")
        added.append(row)
    if added:
        write_csv(QUESTIONS, FIELDS, rows + added)
    print(f"import: {len(added)} drafts add hue | skipped {skipped} (slot nahi mila ya duplicate) | bad lines {bad}")
    print("Ab har draft ko PDF se match karke status=verified karo (ya rejected).")


def cmd_probe(a):
    chunks = load_corpus()
    for score, c in build_probe(chunks)(a.query, a.k):
        print(f"\n[{score}] {c['file']} p{c['page_start']}-{c['page_end']} clause {c['clause_ids'] or '-'} "
              f"status={c.get('status', '?')}\n   {c['text'][:220].replace(chr(10), ' ')}")
    print("\n(ye sirf lexical helper hai, asli BM25/vector retrieval nahi)")


def cmd_diff(a):
    chunks = load_corpus()
    old, new = resolve(chunks, a.old), resolve(chunks, a.new)
    pairs = diff_pairs([c for c in chunks if c["file"] == old], [c for c in chunks if c["file"] == new], top=a.top)
    print(f"old={old}\nnew={new}\nbadle hue provisions (numbers badle wale pehle): {len(pairs)}\n")
    for d in pairs:
        o, n_ = d["old"], d["new"]
        print(f"ratio {d['ratio']} | numbers old-only {d['only_old']} new-only {d['only_new']}")
        print(f"  OLD p{o['page_start']} clause {o['clause_ids'] or '-'}: {o['text'][:200].replace(chr(10), ' ')}")
        print(f"  NEW p{n_['page_start']} clause {n_['clause_ids'] or '-'}: {n_['text'][:200].replace(chr(10), ' ')}\n")


def cmd_validate(a):
    sources = {r["file"]: r for r in read_csv(META / "sources.csv")}
    status = {r["file"]: r for r in read_csv(META / "doc_status.csv")}
    rels = [r for r in read_csv(META / "relations.csv") if any((v or "").strip() for v in r.values())]
    rows = read_csv(QUESTIONS)
    if not rows:
        print(f"{QUESTIONS} khaali ya missing hai.")
        return
    errors, warns = validate(rows, sources, status, rels, None if a.no_quotes else page_text)
    s = summarize(rows)
    share = doc_share(rows)
    for f, pct in share.items():
        if pct > 20:
            warns.append(f"{f} {pct}% answerable questions mein hai (dabdaba), docs badlo")

    print(f"verified {s['verified']} | draft {s['draft']} | rejected {s['rejected']}")
    ok = True
    for cat, tgt in TARGET_DEV.items():
        have = s["by_category"].get(cat, 0)
        ok &= have >= tgt
        print(f"  {cat:<14} {have:>3} / {tgt}")
    ok &= s["graph_only"] >= GRAPH_ONLY_DEV
    ok &= s["self_authored"] >= MIN_SELF
    print(f"graph_only {s['graph_only']} / {GRAPH_ONLY_DEV} | self-authored {s['self_authored']} / {MIN_SELF}"
          f" | splits {s['by_split']}")
    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\nerrors: {len(errors)} | warnings: {len(warns)} | targets: {'MET' if ok else 'NOT YET'}")
    if errors:
        print("golden.jsonl update nahi hui (pehle errors theek karo).")
        sys.exit(1)

    recs = [to_record(r) for r in rows if (r.get("status") or "").strip() == "verified"]
    if not recs:
        print("Abhi koi verified question nahi, golden.jsonl nahi banayi.")
        return
    EXPORT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs) + "\n", encoding="utf-8")
    corpus_man = META / "corpus_manifest.json"
    corpus_sha = json.loads(corpus_man.read_text(encoding="utf-8")).get("corpus_sha", "") if corpus_man.exists() else ""
    man = {"n_questions": len(recs), "by_category": s["by_category"], "graph_only": s["graph_only"],
           "self_authored": s["self_authored"], "by_split": s["by_split"],
           "golden_sha": hashlib.sha256(EXPORT.read_bytes()).hexdigest()[:12], "validated_against_corpus_sha": corpus_sha}
    GMAN.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"golden.jsonl: {len(recs)} questions | golden_sha {man['golden_sha']} | corpus_sha {corpus_sha}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_plan)
    sub.add_parser("packets").set_defaults(fn=cmd_packets)
    sub.add_parser("import").set_defaults(fn=cmd_import)
    p = sub.add_parser("probe")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("diff")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--top", type=int, default=12)
    p.set_defaults(fn=cmd_diff)
    p = sub.add_parser("validate")
    p.add_argument("--no-quotes", action="store_true")
    p.set_defaults(fn=cmd_validate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()