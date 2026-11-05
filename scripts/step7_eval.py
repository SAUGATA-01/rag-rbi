"""Step 7: retrieval eval harness.

  run      retriever ko golden.jsonl par chalao -> eval/runs/<name>/ (metrics.json + per_question.jsonl)
  compare  do runs ka paired comparison (bootstrap CI, wins/losses)
  show     ek run ke failures (jahan gold top-k mein nahi aaya)
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rbi_rag.evalr import KS, MAXK, ChunkIndex, aggregate, group_names, paired, score_question
from rbi_rag.retrievers import REGISTRY, build

META = Path("data/metadata")
CORPUS = Path("data/processed/corpus.jsonl")
GOLD = Path("eval/golden/golden.jsonl")
GMAN = Path("eval/golden/golden_manifest.json")
RUNS = Path("eval/runs")
SUMMARY = Path("eval/results/runs_summary.csv")


def read_jsonl(path):
    if not path.exists():
        print(f"{path} nahi mili.")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(x, pct=True):
    if x is None:
        return "  -  "
    return f"{100 * x:5.1f}" if pct else f"{x:5.2f}"


def cmd_run(a):
    if a.split in ("test", "all") and not a.allow_test:
        print("test split tuning ke dauraan nahi chhooni. Sirf final ablation mein --allow-test lagao.")
        sys.exit(1)
    chunks, golden = read_jsonl(CORPUS), read_jsonl(GOLD)
    recs = [g for g in golden if a.split == "all" or g["split"] == a.split]
    if not recs:
        print(f"split '{a.split}' mein koi question nahi.")
        sys.exit(1)

    cman = json.loads((META / "corpus_manifest.json").read_text(encoding="utf-8")) if (META / "corpus_manifest.json").exists() else {}
    gman = json.loads(GMAN.read_text(encoding="utf-8")) if GMAN.exists() else {}
    if gman.get("validated_against_corpus_sha") and cman.get("corpus_sha") != gman["validated_against_corpus_sha"]:
        print("WARN golden set kisi purane corpus par validate hua tha. "
              "`python scripts\\step6_golden.py validate` dobara chalao.")

    retr = build(a.retriever, chunks, golden)
    idx = ChunkIndex(chunks)
    per_q = []
    for g in recs:
        hits = retr.retrieve(g["question"], a.k, qid=g["id"])
        per_q.append(score_question(g, hits, idx))
    agg = aggregate(per_q)

    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for q in per_q:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    meta = {"run": a.name, "retriever": a.retriever, "k": a.k, "split": a.split, "n_questions": len(per_q),
            "corpus_sha": cman.get("corpus_sha", ""), "golden_sha": gman.get("golden_sha", ""),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "metrics": agg}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"run '{a.name}' | retriever {a.retriever} | split {a.split} | {len(per_q)} questions "
          f"| corpus_sha {meta['corpus_sha']} | golden_sha {meta['golden_sha']}\n")
    hdr = "group".ljust(20) + "  n  " + " ".join(f"R@{k:<3}" for k in KS) + "  S_all@5  MRR@20  quoteR@5  R@5 95% CI"
    print(hdr)
    for name, m in agg["groups"].items():
        if not m["n"]:
            continue
        ci = m["recall@5_ci"]
        print(f"{name:<20}{m['n']:>3}  " + " ".join(fmt(m[f'recall@{k}']) for k in KS) +
              f"  {fmt(m['success_all@5'])}   {fmt(m['mrr@20'], False)}   {fmt(m['recall_quote@5'])}    "
              f"[{fmt(ci[0])}, {fmt(ci[1])}]")
    ov = agg["groups"]["answerable"]
    print(f"\nsuperseded chunk top-5 mein (answerable): {fmt(ov['superseded_in_top5'])}%")
    o = agg["outdated"]
    if o["n"]:
        print(f"outdated_trap: trap top-5 mein {fmt(o['trap_in_top5'])}% | trap gold se pehle {fmt(o['outdated_first_rate'])}%  (n={o['n']})")
    auc = agg["abstain_auc"]
    print(f"abstention AUC (top-1 score se answerable vs unanswerable alag): {'-' if auc is None else f'{auc:.3f}'}"
          f"  (n_unanswerable={agg['n_unanswerable']}; 0.5 = coin flip)")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    row = {"run": a.name, "retriever": a.retriever, "split": a.split, "n": ov["n"],
           "recall@5": ov["recall@5"], "recall@5_lo": ov["recall@5_ci"][0], "recall@5_hi": ov["recall@5_ci"][1],
           "mrr@20": ov["mrr@20"], "recall_quote@5": ov["recall_quote@5"], "success_all@5": ov["success_all@5"],
           "superseded_in_top5": ov["superseded_in_top5"], "outdated_first_rate": o["outdated_first_rate"],
           "abstain_auc": auc, "corpus_sha": meta["corpus_sha"], "golden_sha": meta["golden_sha"]}
    rows = []
    if SUMMARY.exists():
        with SUMMARY.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run"] != a.name]
    rows.append({k: ("" if v is None else (round(v, 4) if isinstance(v, float) else v)) for k, v in row.items()})
    with SUMMARY.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved: {out}  |  summary: {SUMMARY}")


def load_run(name):
    return {q["id"]: q for q in read_jsonl(RUNS / name / "per_question.jsonl")}


def cmd_compare(a):
    A, B = load_run(a.a), load_run(a.b)
    ids = sorted(set(A) & set(B))
    if not ids:
        print("dono runs mein koi common question nahi.")
        sys.exit(1)
    print(f"{a.a}  vs  {a.b}   (diff = {a.b} - {a.a}, common questions: {len(ids)})\n")
    pa = [A[i] for i in ids]
    pb = [B[i] for i in ids]
    ga, gb = group_names(pa), group_names(pb)
    metrics = (("R@5", lambda q: q["recall"]["5"]), ("MRR@20", lambda q: q["rr_page"]),
               ("quoteR@5", lambda q: q["recall_quote5"]))
    print("group".ljust(20) + "metric".ljust(10) + "A".rjust(7) + "B".rjust(7) + "diff".rjust(8) + "   95% CI        W/L/T   verdict")
    for g in ga:
        qa = {q["id"]: q for q in ga[g]}
        qb = {q["id"]: q for q in gb[g]}
        common = sorted(set(qa) & set(qb))
        if not common:
            continue
        for mname, fn in metrics:
            p = paired([fn(qa[i]) for i in common], [fn(qb[i]) for i in common])
            lo, hi = p["ci"]
            verdict = "B better" if lo > 0 else "A better" if hi < 0 else "no clear difference"
            print(f"{g:<20}{mname:<10}{100 * p['mean_a']:7.1f}{100 * p['mean_b']:7.1f}{100 * p['diff']:+8.1f}"
                  f"   [{100 * lo:+5.1f},{100 * hi:+5.1f}]  {p['wins']}/{p['losses']}/{p['ties']}  {verdict}")
    print("\nNote: ~100 questions par 2-3 point ka fark noise hai. Chhote groups (n<15) ke CI bahut chaude hote hain.")


def cmd_show(a):
    per_q = list(load_run(a.run).values())
    fails = [q for q in per_q if q["category"] != "unanswerable"
             and (q["first_rank_page"] is None or q["first_rank_page"] > a.k)]
    print(f"{a.run}: {len(fails)} answerable questions jahan gold top-{a.k} mein nahi aaya "
          f"(out of {sum(1 for q in per_q if q['category'] != 'unanswerable')})\n")
    for q in fails[:a.n]:
        print(f"[{q['id']}] ({q['category']}{', graph_only' if q['graph_only'] else ''}) {q['question']}")
        print("  gold: " + "; ".join(f"{e['file']} p{e['page']}" for e in q["evidence"]))
        for t in q["top"][:3]:
            print(f"  got : {t['file']} p{t['pages'][0]}-{t['pages'][1]} score {t['score']} [{t['status']}]")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--retriever", required=True, choices=sorted(REGISTRY))
    p.add_argument("--name", required=True, help="run ka naam (folder eval/runs/<name>)")
    p.add_argument("--split", default="dev")
    p.add_argument("--k", type=int, default=MAXK)
    p.add_argument("--allow-test", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("show")
    p.add_argument("run")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n", type=int, default=10)
    p.set_defaults(fn=cmd_show)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()