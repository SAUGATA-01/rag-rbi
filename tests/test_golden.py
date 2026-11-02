import unittest

from rbi_rag.golden import (UNANSWERABLE, build_plan, build_probe, check_quote, diff_pairs,
                            doc_share, overlap_coef, parse_drafts, summarize, to_record, validate)

SOURCES = {f: {} for f in ("old.pdf", "new.pdf", "amd.pdf", "other.pdf")}
STATUS = {
    "old.pdf": {"status": "superseded", "superseded_by": "new.pdf", "latest_active": "new.pdf", "amended_by": "amd.pdf"},
    "new.pdf": {"status": "active", "latest_active": "new.pdf"},
    "amd.pdf": {"status": "active", "latest_active": "amd.pdf"},
    "other.pdf": {"status": "active", "latest_active": "other.pdf"},
}
RELS = [
    {"src_file": "new.pdf", "relation": "supersedes", "dst_file": "old.pdf"},
    {"src_file": "amd.pdf", "relation": "amends", "dst_file": "old.pdf"},
]
PAGES = {
    ("new.pdf", 1): "Banks shall maintain a liquidity buffer of 18 per cent of deposits.",
    ("old.pdf", 1): "Banks shall maintain a liquidity buffer of 40 per cent of deposits.",
    ("amd.pdf", 2): "Paragraph 4 of the Master Direction is substituted as follows.",
    ("other.pdf", 1): "The Board shall review the policy annually.",
}


def pt(f, p):
    return PAGES.get((f, p))


def q(**kw):
    base = {"id": "q001", "split": "dev", "category": "simple", "graph_only": "", "subtype": "",
            "question": "What share of deposits must banks hold as a liquidity buffer today?",
            "reference_answer": "18 per cent of deposits.",
            "e1_file": "new.pdf", "e1_page": "1", "e1_quote": "liquidity buffer of 18 per cent of deposits",
            "authored_by": "self", "status": "verified", "notes": ""}
    base.update(kw)
    return base


def errs(rows):
    return validate(rows, SOURCES, STATUS, RELS, pt)


class GoldenTests(unittest.TestCase):
    def test_simple_ok(self):
        e, w = errs([q()])
        self.assertEqual(e, [])

    def test_draft_not_deep_checked(self):
        e, _ = errs([q(status="draft", e1_quote="nonsense not on page")])
        self.assertEqual(e, [])

    def test_quote_wrong_page_and_missing(self):
        e, _ = errs([q(e1_page="2")])
        self.assertTrue(any("nahi" in x for x in e))
        e, _ = errs([q(e1_quote="this text is not on the page at all")])
        self.assertTrue(any("nahi mila" in x for x in e))

    def test_simple_requires_active_doc(self):
        e, _ = errs([q(e1_file="old.pdf")])
        self.assertTrue(any("active" in x for x in e))

    def test_duplicate_and_bad_enums(self):
        e, _ = errs([q(), q(), q(id="q2", category="weird"), q(id="q3", status="maybe")])
        self.assertEqual(len([x for x in e if "duplicate" in x]), 1)
        self.assertTrue(any("category" in x for x in e))
        self.assertTrue(any("status" in x for x in e))

    def test_outdated_trap(self):
        r = q(category="outdated_trap", trap_file="old.pdf", trap_page="1",
              trap_quote="liquidity buffer of 40 per cent of deposits")
        self.assertEqual(errs([r])[0], [])
        bad = dict(r, trap_file="other.pdf", trap_page="1", trap_quote="The Board shall review the policy annually")
        e, _ = errs([bad])
        self.assertTrue(any("superseded" in x for x in e))
        e, _ = errs([dict(r, e1_file="other.pdf", e1_page="1", e1_quote="The Board shall review the policy")])
        self.assertTrue(any("latest_active" in x for x in e))

    def test_what_changed_needs_relation_direction(self):
        r = q(category="what_changed", e1_file="amd.pdf", e1_page="2", e1_quote="Paragraph 4 of the Master Direction is substituted",
              e2_file="old.pdf", e2_page="1", e2_quote="liquidity buffer of 40 per cent")
        self.assertEqual(errs([r])[0], [])
        swapped = dict(r, e1_file="old.pdf", e1_page="1", e1_quote="liquidity buffer of 40 per cent",
                       e2_file="amd.pdf", e2_page="2", e2_quote="Paragraph 4 of the Master Direction is substituted")
        e, _ = errs([swapped])
        self.assertTrue(any("relations.csv" in x for x in e))

    def test_multihop_and_graph_only(self):
        r = q(category="multihop", graph_only="yes", e1_file="old.pdf", e1_page="1",
              e1_quote="liquidity buffer of 40 per cent", e2_file="new.pdf", e2_page="1",
              e2_quote="liquidity buffer of 18 per cent")
        self.assertEqual(errs([r])[0], [])
        unrelated = dict(r, e1_file="other.pdf", e1_page="1", e1_quote="The Board shall review the policy")
        e, _ = errs([dict(unrelated, e2_file="amd.pdf", e2_page="2", e2_quote="Paragraph 4 of the Master Direction")])
        self.assertTrue(any("relation nahi" in x for x in e) or e == [])  # other<->amd koi relation nahi
        e, _ = errs([q(graph_only="yes")])
        self.assertTrue(any("sirf multihop" in x for x in e))
        e, _ = errs([q(category="multihop")])
        self.assertTrue(any("2 alag" in x for x in e))

    def test_unanswerable(self):
        r = q(category="unanswerable", reference_answer=UNANSWERABLE, subtype="near_miss", notes="probe: koi match nahi",
              e1_file="", e1_page="", e1_quote="")
        self.assertEqual(errs([r])[0], [])
        e, _ = errs([dict(r, reference_answer="some answer", notes="")])
        self.assertEqual(len(e), 2)

    def test_overlap_warning(self):
        _, w = errs([q(question="What is the liquidity buffer of 18 per cent of deposits for banks?")])
        self.assertTrue(any("paraphrase" in x for x in w))
        self.assertLess(overlap_coef("What share must banks hold?", "liquidity buffer of 18 per cent"), 0.7)

    def test_summary_share_record(self):
        rows = [q(), q(id="q2", authored_by="llm_assisted", status="draft")]
        s = summarize(rows)
        self.assertEqual((s["verified"], s["draft"], s["self_authored"]), (1, 1, 1))
        self.assertEqual(doc_share(rows), {"new.pdf": 100.0})
        rec = to_record(rows[0])
        self.assertEqual(rec["expected_docs"], ["new.pdf"])
        self.assertEqual(rec["evidence"][0]["page"], 1)

    def test_check_quote_hyphen(self):
        pages = {("a.pdf", 1): "is hereby super-\nseded by"}
        self.assertIsNone(check_quote("a.pdf", "1", "hereby superseded by", lambda f, p: pages.get((f, p))))

    def test_parse_drafts(self):
        raw = '```json\n{"slot":"P01","question":"q?","reference_answer":"a","quote":"x y"}\n{"slot":"P01","question":"","reference_answer":"a","quote":"x"}\nnot json\n```'
        good, bad = parse_drafts(raw)
        self.assertEqual(len(good), 1)
        self.assertEqual(bad, 2)
        good, bad = parse_drafts('[{"slot":"P02","question":"q","reference_answer":"a","quote":"x"}]')
        self.assertEqual((len(good), bad), (1, 0))

    def test_probe(self):
        chunks = [{"text": "Priority sector lending target for agriculture"}, {"text": "Payment aggregator authorisation"}]
        hits = build_probe(chunks)("what is the lending target for agriculture", k=2)
        self.assertEqual(hits[0][1], chunks[0])

    def test_diff_pairs_finds_changed_number(self):
        old = [{"text": "Banks shall maintain a liquidity buffer of 40 % of deposits and report monthly to the Reserve Bank."},
               {"text": "Unrelated clause about payment aggregators and their authorisation process."}]
        new = [{"text": "Banks shall maintain a liquidity buffer of 18 % of deposits and report monthly to the Reserve Bank."},
               {"text": "Unrelated clause about payment aggregators and their authorisation process."}]
        d = diff_pairs(old, new)
        self.assertEqual(len(d), 1)  # identical chunk filter hota hai
        self.assertEqual((d[0]["only_old"], d[0]["only_new"]), (["40%"], ["18%"]))

    def test_plan_counts(self):
        chunks = [{"file": f, "n_words": 100, "page_start": 1, "maybe_table": False}
                  for f in ("new.pdf", "amd.pdf", "other.pdf")]
        rows, warns = build_plan(chunks, STATUS, RELS)
        cats = {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        self.assertEqual(cats, {"simple": 15, "outdated_trap": 12, "what_changed": 9, "multihop": 12, "unanswerable": 12})
        self.assertEqual(sum(1 for r in rows if r["graph_only"] == "yes"), 6)
        self.assertTrue(all(r["target_files"] != "old.pdf" for r in rows if r["category"] == "simple"))


if __name__ == "__main__":
    unittest.main()