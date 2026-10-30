
import unittest

from rbi_rag.chunker import ChunkConfig, chunk_pages, clean_pages

HDR = "Reserve Bank of India | Test Directions 2025"
PAGES = [
    f"{HDR}\nChapter I \u2013 Preliminary\n1. Short title and commencement\n"
    "These Directions shall be called the Test Directions, 2025 and shall come into effect immediately.\n"
    "2. Applicability\n(a) These Directions apply to all banks.\n"
    "(b) These Directions do not apply to payment banks.\nPage 1 of 4",
    f"{HDR}\n3. Definitions\nIn these Directions, unless the context otherwise requires, "
    "the terms defined below shall have the meanings assigned to them.\n"
    "Chapter II \u2013 Requirements\n4. Reporting\n"
    "Banks shall report to the Reserve Bank within 30 days.\n"
    "25 June 2025 is the effective date of this reporting requirement.\nPage 2 of 4",
    f"{HDR}\n5.1 Review\nThe Board shall review the policy annually.\n"
    "5.2 Repeal\nThe earlier Master Direction dated 4 September 2020 is hereby repealed.\nPage 3 of 4",
    f"{HDR}\n6. Interpretation\nIf any doubt arises, the decision of the Reserve Bank shall be final.\nPage 4 of 4",
]
CFG = ChunkConfig(min_words=5, max_words=50)


class ChunkerTests(unittest.TestCase):
    def test_noise_removed(self):
        lines = clean_pages(PAGES, CFG)
        flat = [l for p in lines for l in p]
        self.assertNotIn(HDR, flat)
        self.assertFalse(any(l.startswith("Page ") for l in flat))

    def test_clauses_headings_pages(self):
        ch = chunk_pages(PAGES, "t", cfg=CFG)
        self.assertEqual([c["clause_ids"] for c in ch], ["1", "2", "3", "4", "5.1", "5.2", "6"])
        self.assertEqual([c["heading"] for c in ch[:3]], ["Chapter I \u2013 Preliminary"] * 3)
        self.assertEqual([c["heading"] for c in ch[3:]], ["Chapter II \u2013 Requirements"] * 4)
        self.assertEqual([(c["page_start"], c["page_end"]) for c in ch],
                         [(1, 1), (1, 1), (2, 2), (2, 2), (3, 3), (3, 3), (4, 4)])

    def test_date_line_is_not_a_clause(self):
        ch = chunk_pages(PAGES, "t", cfg=CFG)
        self.assertNotIn("25", [c["clause_ids"] for c in ch])
        self.assertIn("25 June 2025 is the effective date", ch[3]["text"])  # inside clause 4

    def test_subclauses_stay_inside_parent(self):
        ch = chunk_pages(PAGES, "t", cfg=CFG)
        c2 = ch[1]["text"]
        self.assertTrue(c2.startswith("2. Applicability\n(a) "))
        self.assertIn("\n(b) ", c2)

    def test_split_long_clause_at_subclauses(self):
        subs = "\n".join(f"({chr(97 + i)}) item number {i} has exactly eight words in it" for i in range(10))
        page = f"Chapter I \u2013 Test\n7. Long clause intro\n{subs}"
        ch = chunk_pages([page], "t", cfg=ChunkConfig(min_words=5, max_words=30))
        self.assertGreater(len(ch), 1)
        self.assertTrue(all(c["n_words"] <= 30 for c in ch))
        self.assertTrue(all(c["clause_ids"] == "7" for c in ch))
        self.assertTrue(all(c["text"].startswith("(") for c in ch[1:]))  # split at sub-clause boundaries

    def test_merge_small_within_heading(self):
        page = ("Chapter I \u2013 Test\n8. Scope\n9. Applicability of these rules to banks and others here\n"
                "10. Next clause with enough words to stand alone fine")
        ch = chunk_pages([page], "t", cfg=ChunkConfig(min_words=6, max_words=50))
        self.assertEqual([c["clause_ids"] for c in ch], ["8, 9", "10"])

    def test_no_merge_across_headings(self):
        page = "Chapter I \u2013 A\n1. Tiny\nChapter II \u2013 B\n2. Another tiny"
        ch = chunk_pages([page], "t", cfg=ChunkConfig(min_words=6, max_words=50))
        self.assertEqual([c["heading"] for c in ch], ["Chapter I \u2013 A", "Chapter II \u2013 B"])


    def test_dashed_annex_headings_stay_separate(self):
        page = (
            "Annex - I A\n"
            "List of Districts with comparatively high PSL credit\n"
            "District Alpha District Beta District Gamma District Delta\n"
            "Annex \u2013 I B\n"
            "List of Districts with comparatively low PSL credit\n"
            "District Epsilon District Zeta District Eta District Theta\n"
            "Annex \u2013 II\n"
            "Indicative list of eligible activities under Agriculture Infrastructure"
        )
        ch = chunk_pages([page], "annex_test", cfg=ChunkConfig(min_words=2, max_words=50))
        self.assertEqual([c["heading"] for c in ch], ["Annex - I A", "Annex \u2013 I B", "Annex \u2013 II"])
        self.assertIn("high PSL credit", ch[0]["text"])
        self.assertNotIn("low PSL credit", ch[0]["text"])
        self.assertIn("low PSL credit", ch[1]["text"])
        self.assertNotIn("high PSL credit", ch[1]["text"])



if __name__ == "__main__":
    unittest.main()
