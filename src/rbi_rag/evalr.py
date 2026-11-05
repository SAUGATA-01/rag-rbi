"""Step 7: retrieval evaluation (LLM ke bina). Gold = (file, page, quote), chunk-agnostic."""
from __future__ import annotations

import random

from rbi_rag.relations import norm

KS = (1, 3, 5, 10, 20)
MAXK = 20
ANSWERABLE = ("simple", "multihop", "outdated_trap", "what_changed")


# ---------------- matching ----------------
class ChunkIndex:
    """chunk text ka normalized version cache karta hai (quote matching ke liye)."""

    def __init__(self, chunks):
        self._norm = {}

    def norm_text(self, hit):
        cid = hit["chunk_id"]
        if cid not in self._norm:
            self._norm[cid] = norm(hit["text"])
        return self._norm[cid]


def page_hit(hit, file, page) -> bool:
    """Lenient: retrieved chunk ka page-range gold page ko cover karta hai."""
    return hit["file"] == file and hit["page_start"] <= page <= hit["page_end"]


def quote_hit(hit, file, quote, idx) -> bool:
    """Strict: gold quote chunk ke text ke andar hai."""
    return hit["file"] == file and norm(quote) in idx.norm_text(hit)


def first_rank(hits, pred):
    for r, h in enumerate(hits, start=1):
        if pred(h):
            return r
    return None


# ---------------- per-question scoring ----------------
def recall_at(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else None


def success_all(ranks, k):
    return float(bool(ranks) and all(r is not None and r <= k for r in ranks))


def rr(rank, maxk=MAXK):
    return 1.0 / rank if rank and rank <= maxk else 0.0


def score_question(rec, hits, idx):
    ev = rec.get("evidence", [])
    rp = [first_rank(hits, lambda h, e=e: page_hit(h, e["file"], e["page"])) for e in ev]
    rq = [first_rank(hits, lambda h, e=e: quote_hit(h, e["file"], e["quote"], idx)) for e in ev]
    trap_rank = None
    if rec.get("trap"):
        t = rec["trap"]
        trap_rank = first_rank(hits, lambda h: page_hit(h, t["file"], t["page"]))
    fp = min([r for r in rp if r], default=None)
    fq = min([r for r in rq if r], default=None)
    out = {
        "id": rec["id"], "category": rec["category"], "graph_only": bool(rec.get("graph_only")),
        "split": rec.get("split", ""), "question": rec["question"],
        "evidence": [{"file": e["file"], "page": e["page"]} for e in ev],
        "ranks_page": rp, "ranks_quote": rq, "trap_rank": trap_rank,
        "first_rank_page": fp, "first_rank_quote": fq,
        "recall": {str(k): recall_at(rp, k) for k in KS},
        "recall_quote5": recall_at(rq, 5),
        "success_all5": success_all(rp, 5),
        "rr_page": rr(fp), "rr_quote": rr(fq),
        "superseded_in_top5": any(h.get("status") == "superseded" for h in hits[:5]),
        "top1_score": float(hits[0].get("score", 0.0)) if hits else 0.0,
        "top": [{"chunk_id": h["chunk_id"], "file": h["file"], "pages": [h["page_start"], h["page_end"]],
                 "score": round(float(h.get("score", 0.0)), 4), "status": h.get("status", "")} for h in hits[:10]],
    }
    out["outdated_first"] = (trap_rank is not None and (fp is None or trap_rank < fp)) if rec.get("trap") else None
    return out


# ---------------- statistics ----------------
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def bootstrap_ci(values, n=1000, seed=7, alpha=0.05):
    vals = list(values)
    if not vals:
        return (None, None)
    if len(vals) == 1:
        return (vals[0], vals[0])
    rnd = random.Random(seed)
    m = len(vals)
    means = sorted(sum(vals[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return (means[int(n * alpha / 2)], means[min(n - 1, int(n * (1 - alpha / 2)))])


def auc(pos, neg):
    """P(pos score > neg score), ties = 0.5. Scale-free, isliye alag retrievers ke beech comparable."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


def group_names(per_q):
    ans = [q for q in per_q if q["category"] in ANSWERABLE]
    g = {"answerable": ans}
    for c in ANSWERABLE:
        g[c] = [q for q in ans if q["category"] == c]
    g["multihop_graph_only"] = [q for q in ans if q["category"] == "multihop" and q["graph_only"]]
    g["multihop_text"] = [q for q in ans if q["category"] == "multihop" and not q["graph_only"]]
    return g


def group_metrics(qs):
    if not qs:
        return {"n": 0}
    r5 = [q["recall"]["5"] for q in qs]
    mrr = [q["rr_page"] for q in qs]
    lo5, hi5 = bootstrap_ci(r5)
    lom, him = bootstrap_ci(mrr)
    m = {"n": len(qs)}
    for k in KS:
        m[f"recall@{k}"] = mean(q["recall"][str(k)] for q in qs)
    m.update({"success_all@5": mean(q["success_all5"] for q in qs), "mrr@20": mean(mrr),
              "recall_quote@5": mean(q["recall_quote5"] for q in qs),
              "mrr_quote@20": mean(q["rr_quote"] for q in qs),
              "recall@5_ci": [lo5, hi5], "mrr@20_ci": [lom, him],
              "superseded_in_top5": mean(float(q["superseded_in_top5"]) for q in qs)})
    return m


def aggregate(per_q):
    out = {"groups": {name: group_metrics(qs) for name, qs in group_names(per_q).items()}}
    ans = [q["top1_score"] for q in per_q if q["category"] in ANSWERABLE]
    unans = [q["top1_score"] for q in per_q if q["category"] == "unanswerable"]
    out["abstain_auc"] = auc(ans, unans)
    out["n_unanswerable"] = len(unans)
    traps = [q for q in per_q if q["category"] == "outdated_trap" and q["outdated_first"] is not None]
    out["outdated"] = {
        "n": len(traps),
        "outdated_first_rate": mean(float(q["outdated_first"]) for q in traps),
        "trap_in_top5": mean(float(q["trap_rank"] is not None and q["trap_rank"] <= 5) for q in traps),
    }
    return out


def paired(a_vals, b_vals, seed=7):
    """b - a (paired). Mean diff, 95% bootstrap CI, wins/losses/ties."""
    diffs = [b - a for a, b in zip(a_vals, b_vals)]
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return {"n": len(diffs), "mean_a": mean(a_vals), "mean_b": mean(b_vals), "diff": mean(diffs),
            "ci": [lo, hi], "wins": sum(1 for d in diffs if d > 1e-12), "losses": sum(1 for d in diffs if d < -1e-12),
            "ties": sum(1 for d in diffs if abs(d) <= 1e-12)}