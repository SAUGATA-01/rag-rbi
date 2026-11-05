import math
import tempfile
import unittest

import numpy as np

from rbi_rag.bm25 import BM25
from rbi_rag.evalr import ChunkIndex, score_question
from rbi_rag.retrievers import BM25Retriever, VectorRetriever
from rbi_rag.textproc import stem, tokenize
from rbi_rag.vector import EmbeddingCache, HashEmbedder, VectorIndex, text_key


class TextProcTests(unittest.TestCase):
    def test_stem_unifies_forms(self):
        self.assertEqual(stem("issued"), stem("issue"))
        self.assertEqual(stem("issues"), stem("issue"))
        self.assertEqual(stem("lending"), stem("lend"))
        self.assertEqual(stem("banks"), "bank")
        self.assertEqual(stem("policies"), stem("policy"))
        self.assertEqual(stem("2025"), "2025")          # numbers jaise ke taise
        self.assertEqual(stem("business"), "business")  # -ss nahi katta

    def test_tokenize(self):
        self.assertEqual(tokenize("The Banks shall report 18% of deposits."), ["bank", "report", "18", "deposit"])
        self.assertEqual(tokenize("Banks", use_stem=False), ["banks"])


class BM25Tests(unittest.TestCase):
    DOCS = [["bank", "loan", "rate"], ["bank", "deposit"], ["loan", "loan", "interest"]]

    def test_three_document_example(self):
        # haath se nikala hua: N=3, avgdl=8/3, k1=1.5, b=0.75, idf(loan)=idf(bank)=ln(1.6)
        bm = BM25(self.DOCS)
        s = bm.scores(["loan", "bank"])
        self.assertAlmostEqual(bm.idf["loan"], math.log(1.6), places=6)
        self.assertAlmostEqual(s[0], 0.889948, places=5)   # d1: dono terms
        self.assertAlmostEqual(s[1], 0.529582, places=5)   # d2: sirf bank, chhota doc
        self.assertAlmostEqual(s[2], 0.645499, places=5)   # d3: loan tf=2
        self.assertEqual([i for i, _ in bm.top(["loan", "bank"], 3)], [0, 2, 1])

    def test_matches_independent_formula(self):
        bm = BM25(self.DOCS, k1=1.2, b=0.5)
        N, avg = 3, 8 / 3
        for qi, q in enumerate([["loan"], ["rate", "interest"], ["bank", "deposit", "loan"]]):
            for d, doc in enumerate(self.DOCS):
                exp = 0.0
                for t in set(q):
                    df = sum(1 for x in self.DOCS if t in x)
                    tf = doc.count(t)
                    if tf:
                        exp += math.log(1 + (N - df + .5) / (df + .5)) * tf * 2.2 / (tf + 1.2 * (0.5 + 0.5 * len(doc) / avg))
                self.assertAlmostEqual(bm.scores(q)[d], exp, places=9)

    def test_edge_cases(self):
        bm = BM25(self.DOCS)
        self.assertEqual(bm.top(["unknownterm"], 5), [])
        self.assertEqual(bm.top([], 5), [])
        self.assertEqual(len(bm.top(["bank", "loan", "rate"], 1)), 1)
        self.assertEqual(BM25([]).top(["x"], 3), [])
        tie = BM25([["a", "b"], ["a", "b"]])
        self.assertEqual([i for i, _ in tie.top(["a"], 2)], [0, 1])   # tie: chhota index pehle

    def test_repeated_query_term_counted_once(self):
        bm = BM25(self.DOCS)
        self.assertTrue(np.allclose(bm.scores(["loan"]), bm.scores(["loan", "loan"])))


class VectorTests(unittest.TestCase):
    def test_hash_embedder_normalized_deterministic(self):
        e = HashEmbedder()
        a = e.encode_passages(["liquidity buffer of banks", "payment aggregator"])
        b = HashEmbedder().encode_passages(["liquidity buffer of banks", "payment aggregator"])
        self.assertTrue(np.allclose(a, b))
        self.assertTrue(np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5))

    def test_cache_incremental(self):
        with tempfile.TemporaryDirectory() as d:
            texts = ["alpha beta", "gamma delta", "epsilon zeta"]
            e = HashEmbedder()
            m1, s1 = EmbeddingCache(d, e.name).get_matrix(texts, e)
            self.assertEqual((s1["embedded"], s1["cached"]), (3, 0))
            e2 = HashEmbedder()
            m2, s2 = EmbeddingCache(d, e2.name).get_matrix(texts, e2)
            self.assertEqual((s2["embedded"], s2["cached"], e2.calls), (0, 3, []))   # embedder chala hi nahi
            self.assertTrue(np.allclose(m1, m2))
            texts2 = ["alpha beta", "gamma delta CHANGED", "epsilon zeta", "eta theta"]
            e3 = HashEmbedder()
            m3, s3 = EmbeddingCache(d, e3.name).get_matrix(texts2, e3)
            self.assertEqual((s3["embedded"], s3["cached"], e3.calls), (2, 2, [2]))
            self.assertTrue(np.allclose(m3[0], m1[0]) and np.allclose(m3[2], m1[2]))   # alignment sahi
            # orphan (purana 'gamma delta') cache se hat gaya
            c = EmbeddingCache(d, e3.name)
            self.assertNotIn(text_key("gamma delta"), c.vecs)
            self.assertEqual(len(c.vecs), 4)

    def test_cache_dim_mismatch_resets(self):
        with tempfile.TemporaryDirectory() as d:
            e = HashEmbedder(dim=64)
            EmbeddingCache(d, "same-name").get_matrix(["a b c"], e)
            e2 = HashEmbedder(dim=128)
            m, s = EmbeddingCache(d, "same-name").get_matrix(["a b c"], e2)
            self.assertEqual((m.shape, s["embedded"]), ((1, 128), 1))

    def test_index_top_and_ties(self):
        idx = VectorIndex(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32))
        top = idx.top(np.array([1, 0], dtype=np.float32), 3)
        self.assertEqual([i for i, _ in top], [0, 2, 1])    # tie: chhota index pehle


CHUNKS = [
    {"chunk_id": "a::0", "file": "a.pdf", "page_start": 1, "page_end": 1, "status": "active",
     "text": "Banks shall maintain a liquidity buffer of 18 per cent of deposits.",
     "embed_text": "Prudential Norms | Clause 1\nBanks shall maintain a liquidity buffer of 18 per cent of deposits."},
    {"chunk_id": "a::1", "file": "a.pdf", "page_start": 2, "page_end": 2, "status": "active",
     "text": "Payment aggregators require authorisation from the Reserve Bank.",
     "embed_text": "Payments Rules | Clause 2\nPayment aggregators require authorisation from the Reserve Bank."},
    {"chunk_id": "b::0", "file": "b.pdf", "page_start": 1, "page_end": 1, "status": "superseded",
     "text": "Kisan credit card limits for agriculture loans are reviewed annually.",
     "embed_text": "Kisan Credit | Clause 1\nKisan credit card limits for agriculture loans are reviewed annually."},
]
REC = {"id": "q1", "category": "simple", "graph_only": False, "split": "dev", "question": "x",
       "evidence": [{"file": "a.pdf", "page": 2, "quote": "Payment aggregators require authorisation"}], "trap": None}


class RetrieverTests(unittest.TestCase):
    def test_bm25_retriever_end_to_end_with_harness(self):
        r = BM25Retriever(CHUNKS)
        hits = r.retrieve("which payment aggregator needs authorization?", 5)
        self.assertEqual(hits[0]["chunk_id"], "a::1")
        self.assertGreater(hits[0]["score"], 0)
        s = score_question(REC, hits, ChunkIndex(CHUNKS))
        self.assertEqual(s["first_rank_page"], 1)

    def test_bm25_text_field_vs_embed_text(self):
        # 'Prudential' sirf embed_text (title) mein hai, text mein nahi
        self.assertEqual(BM25Retriever(CHUNKS, field="embed_text").retrieve("Prudential", 5)[0]["chunk_id"], "a::0")
        self.assertEqual(BM25Retriever(CHUNKS, field="text").retrieve("Prudential", 5), [])

    def test_vector_retriever_with_hash_embedder(self):
        with tempfile.TemporaryDirectory() as d:
            r = VectorRetriever(CHUNKS, embedder=HashEmbedder(), cache_root=d)
            self.assertEqual(r.stats["embedded"], 3)
            hits = r.retrieve("kisan credit card agriculture loans", 2)
            self.assertEqual(hits[0]["chunk_id"], "b::0")
            r2 = VectorRetriever(CHUNKS, embedder=HashEmbedder(), cache_root=d)
            self.assertEqual((r2.stats["embedded"], r2.stats["cached"]), (0, 3))


if __name__ == "__main__":
    unittest.main()