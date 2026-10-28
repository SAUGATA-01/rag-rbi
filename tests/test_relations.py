
import unittest

from rbi_rag.relations import derive_status, find_candidates, norm, validate

SOURCES = {
    "A_old.pdf": {"date": "2020-09-04"},
    "B_mid.pdf": {"date": "2023-01-10"},
    "C_new.pdf": {"date": "2025-06-25"},
    "D_amd.pdf": {"date": "2024-03-01"},
}
PAGES = {
    ("B_mid.pdf", 1): "Preamble text.",
    ("B_mid.pdf", 2): "The Master Direction dated Sep 04, 2020 is hereby super-\nseded by these Directions.",
    ("C_new.pdf", 3): "These Directions repeal the earlier \u201cMaster Direction\u201d of 2023.",
}


def page_text(f, p):
    return PAGES.get((f, p))


def row(**kw):
    base = {"src_file": "B_mid.pdf", "relation": "supersedes", "dst_file": "A_old.pdf",
            "dst_ref": "Master Direction dated Sep 04, 2020", "evidence_source": "pdf_text",
            "evidence_page": "2", "evidence_quote": "is hereby superseded by these Directions",
            "found_by": "manual", "notes": ""}
    base.update(kw)
    return base


class RelationTests(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(norm("super-\nseded  By\u00a0\u201cX\u201d."), "superseded by x")

    def test_valid_row_passes_with_hyphen_linebreak(self):
        errs, warns = validate([row()], SOURCES, page_text)
        self.assertEqual(errs, [])

    def test_quote_on_neighbor_page_is_reported(self):
        errs, _ = validate([row(evidence_page="1")], SOURCES, page_text)
        self.assertEqual(len(errs), 1)
        self.assertIn("found on page 2", errs[0])

    def test_quote_missing(self):
        errs, _ = validate([row(evidence_quote="this sentence does not exist anywhere")], SOURCES, page_text)
        self.assertIn("not found", errs[0])

    def test_bad_rows(self):
        errs, _ = validate([row(src_file="nope.pdf"), row(relation="kills"),
                            row(dst_file="ghost.pdf"), row(dst_ref=" ")], SOURCES, page_text)
        self.assertEqual(len(errs), 4)

    def test_blank_rows_ignored(self):
        blank = {k: "" for k in row()}
        errs, warns = validate([blank, row(), blank], SOURCES, page_text)
        self.assertEqual((errs, warns), ([], []))

    def test_external_target_allowed(self):
        errs, _ = validate([row(dst_file="", dst_ref="Circular of 2019 not in corpus")], SOURCES, page_text)
        self.assertEqual(errs, [])

    def test_cycle_detected(self):
        r1 = row(src_file="B_mid.pdf", dst_file="A_old.pdf")
        r2 = row(src_file="A_old.pdf", dst_file="B_mid.pdf", evidence_source="rbi_web", notes="http://x")
        errs, _ = validate([r1, r2], SOURCES)
        self.assertTrue(any("cycle" in e for e in errs))

    def test_date_order_warning(self):
        r = row(src_file="A_old.pdf", dst_file="C_new.pdf", evidence_source="rbi_web", notes="http://x")
        _, warns = validate([r], SOURCES)
        self.assertTrue(any("is older than" in w for w in warns))

    def test_status_chain_and_amend(self):
        rels = [row(src_file="B_mid.pdf", dst_file="A_old.pdf"),
                row(src_file="C_new.pdf", dst_file="B_mid.pdf"),
                row(src_file="D_amd.pdf", relation="amends", dst_file="B_mid.pdf")]
        st = derive_status(rels, SOURCES)
        self.assertEqual(st["A_old.pdf"]["status"], "superseded")
        self.assertEqual(st["B_mid.pdf"]["status"], "superseded")
        self.assertEqual(st["C_new.pdf"]["status"], "active")
        self.assertEqual(st["D_amd.pdf"]["status"], "active")
        self.assertEqual(st["A_old.pdf"]["latest_active"], "C_new.pdf")  # follow the chain
        self.assertEqual(st["B_mid.pdf"]["amended_by"], "D_amd.pdf")     # amendment does not change status

    def test_withdrawn(self):
        st = derive_status([row(src_file="C_new.pdf", relation="withdraws", dst_file="A_old.pdf")], SOURCES)
        self.assertEqual(st["A_old.pdf"]["status"], "withdrawn")
        self.assertEqual(st["A_old.pdf"]["latest_active"], "")

    def test_candidate_finder(self):
        chunks = [{"file": "x.pdf", "chunk_id": "x::0001", "page_start": 5,
                   "text": "Banks shall report monthly. The earlier circular is hereby repealed.\nOther text."}]
        c = find_candidates(chunks)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]["triggers"], "repeal")
        self.assertEqual(c[0]["page"], 5)


if __name__ == "__main__":
    unittest.main()
