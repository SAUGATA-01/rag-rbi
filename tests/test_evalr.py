import unittest

from rbi_rag.evalr import (ChunkIndex, aggregate, auc, bootstrap_ci, page_hit, paired, quote_hit,
                           rr, score_question)
from rbi_rag.retrievers import IdfOverlapRetriever, OracleRetriever, RandomRetriever


def chunk(i, file, p0, p1, text, status="active"):
    return {"chunk_id": f"{file}::{i}", "file": file, "page_start": p0, "page_end": p1,
            "text": text, "status": status}


CHUNKS = [
    chunk(0, "new.pdf", 1, 1, "Banks shall maintain a liquidity buffer of 18 per cent of deposits."),
    chunk(1, "new.pdf", 2, 3, "Reporting is monthly to the Reserve Bank of India."),
    chunk(2, "old.pdf", 1, 1, "Banks shall maintain a liquidity buffer of 40 per cent of deposits.", "superseded"),
    chunk(3, "other.pdf", 1, 1, "Payment aggregators require authorisation."),
]
GOLD = [
    {"id": "q1", "category": "outdated_trap", "graph_only": False, "split": "dev",
     "question": "What liquidity buffer must banks keep?",
     "evidence": [{"file": "new.pdf", "page": 1, "quote": "liquidity buffer of 18 per cent of deposits"}],
     "trap": {"file": "old.pdf", "page": 1, "quote": "liquidity buffer of 40 per cent"}},
    {"id": "q2", "category": "multihop", "graph_only": True, "split": "dev",
     "question": "Which documents cover buffers and reporting?",
     "evidence": [{"file": "new.pdf", "page": 1, "quote": "liquidity buffer of 18 per cent"},
                  {"file": "new.pdf", "page": 3, "quote": "monthly to the Reserve Bank"}], "trap": None},
    {"id": "q3", "category": "unanswerable", "graph_only": False, "split": "dev",
     "question": "What is the GST rate on cars?", "evidence": [], "trap": None},
]
IDX = ChunkIndex(CHUNKS)


class EvalTests(unittest.TestCase):
    def test_hits(self):
        self.assertTrue(page_hit(CHUNKS[1], "new.pdf", 3))        # range [2,3]
        self.assertFalse(page_hit(CHUNKS[1], "new.pdf", 4))
        self.assertFalse(page_hit(CHUNKS[1], "old.pdf", 2))
        self.assertTrue(quote_hit(CHUNKS[0], "new.pdf", "Liquidity BUFFER of 18 per cent", IDX))
        self.assertFalse(quote_hit(CHUNKS[2], "new.pdf", "liquidity buffer of 40", IDX))  # galat file

    def test_score_question_ranks(self):
        hits = [dict(CHUNKS[2], score=2.0), dict(CHUNKS[0], score=1.0), dict(CHUNKS[3], score=0.5)]
        s = score_question(GOLD[0], hits, IDX)
        self.assertEqual(s["first_rank_page"], 2)
        self.assertEqual(s["trap_rank"], 1)
        self.assertTrue(s["outdated_first"])               # trap gold se pehle aaya
        self.assertTrue(s["superseded_in_top5"])
        self.assertEqual(s["rr_page"], 0.5)
        self.assertEqual(s["recall"]["1"], 0.0)
        self.assertEqual(s["recall"]["5"], 1.0)

    def test_multi_evidence_recall(self):
        hits = [dict(CHUNKS[0], score=1.0), dict(CHUNKS[3], score=0.9)]
        s = score_question(GOLD[1], hits, IDX)
        self.assertEqual(s["ranks_page"], [1, None])
        self.assertEqual(s["recall"]["5"], 0.5)
        self.assertEqual(s["success_all5"], 0.0)
        hits.append(dict(CHUNKS[1], score=0.8))
        s = score_question(GOLD[1], hits, IDX)
        self.assertEqual(s["ranks_page"], [1, 3])
        self.assertEqual(s["success_all5"], 1.0)

    def test_rr_cap(self):
        self.assertEqual(rr(None), 0.0)
        self.assertEqual(rr(21), 0.0)
        self.assertEqual(rr(4), 0.25)

    def test_auc(self):
        self.assertEqual(auc([3, 2], [1]), 1.0)
        self.assertEqual(auc([1], [3]), 0.0)
        self.assertEqual(auc([1, 1], [1]), 0.5)
        self.assertIsNone(auc([], [1]))

    def test_bootstrap_deterministic_and_bounds(self):
        v = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        a, b = bootstrap_ci(v), bootstrap_ci(v)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], sum(v) / len(v))
        self.assertGreaterEqual(a[1], sum(v) / len(v))
        self.assertEqual(bootstrap_ci([0.5]), (0.5, 0.5))

    def test_oracle_is_perfect_and_random_is_not(self):
        idx = ChunkIndex(CHUNKS)
        for mode in ("page", "quote"):
            o = OracleRetriever(CHUNKS, GOLD, mode)
            per_q = [score_question(g, o.retrieve(g["question"], 20, qid=g["id"]), idx) for g in GOLD]
            ag = aggregate(per_q)["groups"]["answerable"]
            self.assertEqual(ag["recall@5"], 1.0, mode)
            self.assertEqual(ag["mrr@20"], 1.0, mode)
        r = RandomRetriever(CHUNKS, GOLD)
        a1 = r.retrieve("x", 3, qid="q1")
        a2 = r.retrieve("x", 3, qid="q1")
        self.assertEqual([h["chunk_id"] for h in a1], [h["chunk_id"] for h in a2])  # seeded

    def test_idf_baseline_finds_buffer_chunk(self):
        hits = IdfOverlapRetriever(CHUNKS).retrieve("liquidity buffer deposits banks", 3)
        self.assertIn(hits[0]["chunk_id"], ("new.pdf::0", "old.pdf::2"))

    def test_aggregate_groups_and_abstain(self):
        idx = ChunkIndex(CHUNKS)
        per_q = []
        for g in GOLD:
            hits = [dict(CHUNKS[0], score=5.0 if g["category"] != "unanswerable" else 1.0)]
            per_q.append(score_question(g, hits, idx))
        ag = aggregate(per_q)
        self.assertEqual(ag["abstain_auc"], 1.0)               # answerable ka top1 score zyada
        self.assertEqual(ag["groups"]["multihop_graph_only"]["n"], 1)
        self.assertEqual(ag["groups"]["answerable"]["n"], 2)
        self.assertEqual(ag["outdated"]["n"], 1)

    def test_paired(self):
        p = paired([0.0, 0.0, 1.0, 0.5], [1.0, 0.5, 1.0, 0.0])
        self.assertEqual((p["wins"], p["losses"], p["ties"]), (2, 1, 1))
        self.assertAlmostEqual(p["diff"], (1.0 + 0.5 + 0.0 - 0.5) / 4)


if __name__ == "__main__":
    unittest.main()