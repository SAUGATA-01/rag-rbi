import json
import tempfile
import unittest
from pathlib import Path

from rbi_rag.chunker import ChunkConfig
from rbi_rag.ingest import (cache_valid, cfg_key, corpus_manifest, doc_report, enrich,
                            sha256_file, text_hash)


def ch(words=100, clause="1", table=False):
    return {"n_words": words, "clause_ids": clause, "maybe_table": table, "text": "x " * words}


class IngestTests(unittest.TestCase):
    def test_text_hash_ignores_space_and_punct(self):
        self.assertEqual(text_hash("The Bank  shall, report."), text_hash("the bank shall report"))
        self.assertNotEqual(text_hash("bank A"), text_hash("bank B"))

    def test_enrich_adds_status_and_hash(self):
        c = enrich({"text": "abc def"}, {"status": "superseded", "latest_active": "N.pdf"})
        self.assertEqual(c["status"], "superseded")
        self.assertEqual(c["latest_active"], "N.pdf")
        self.assertEqual(c["superseded_by"], "")
        self.assertEqual(len(c["text_hash"]), 16)

    def test_enrich_unknown_without_status_row(self):
        self.assertEqual(enrich({"text": "a b"}, None)["status"], "unknown")

    def test_cache_key_changes_with_config_and_parser(self):
        a = cfg_key(ChunkConfig(), "pymupdf")
        self.assertEqual(a, cfg_key(ChunkConfig(), "pymupdf"))
        self.assertNotEqual(a, cfg_key(ChunkConfig(max_words=200), "pymupdf"))
        self.assertNotEqual(a, cfg_key(ChunkConfig(), "pdfplumber"))

    def test_cache_valid(self):
        rec = {"pdf_sha": "p", "cfg_key": "k"}
        self.assertTrue(cache_valid(rec, "p", "k"))
        self.assertFalse(cache_valid(rec, "p2", "k"))
        self.assertFalse(cache_valid(rec, "p", "k2"))
        self.assertFalse(cache_valid(None, "p", "k"))

    def test_doc_report_healthy(self):
        r = doc_report("a.pdf", [ch() for _ in range(20)], pages=10, empty_pages=0)
        self.assertEqual(r["flags"], "")
        self.assertEqual(r["median_words"], 100)

    def test_doc_report_flags(self):
        bad = [ch(words=10, clause="", table=True) for _ in range(4)]
        r = doc_report("b.pdf", bad, pages=40, empty_pages=20)
        for f in ("many_noclause", "many_tables", "tiny_chunks", "few_chunks", "empty_pages"):
            self.assertIn(f, r["flags"])
        self.assertEqual(doc_report("c.pdf", [], pages=3, empty_pages=0)["flags"], "no_chunks")

    def test_manifest_hash_tracks_content(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "c.jsonl").write_text(json.dumps({"a": 1}) + "\n")
            (d / "s.csv").write_text("file\nx.pdf\n")
            m1 = corpus_manifest(d / "c.jsonl", 1, 1, "pymupdf", ChunkConfig(), d / "s.csv", d / "none.csv")
            (d / "c.jsonl").write_text(json.dumps({"a": 2}) + "\n")
            m2 = corpus_manifest(d / "c.jsonl", 1, 1, "pymupdf", ChunkConfig(), d / "s.csv", d / "none.csv")
            self.assertNotEqual(m1["corpus_sha"], m2["corpus_sha"])
            self.assertEqual(m1["status_sha"], "")
            self.assertEqual(sha256_file(d / "s.csv")[:12], m1["sources_sha"])


if __name__ == "__main__":
    unittest.main()