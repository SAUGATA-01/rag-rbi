# RBI-RAG

A from-scratch RAG system over RBI Master Directions and circulars.  
Citation-graph aware, evaluation-first. Research project, not legal or compliance advice.

Status: Step 5 ingestion complete; corpus integrity and repeat-cache checks passed.

## Step 2: Parser comparison (20 RBI PDFs, up to 30 pages each)

| Parser | Text time (s) | Clause lines | Sub-clause lines | Tables found |
|---|---:|---:|---:|---:|
| PyMuPDF | 3.6 | 625 | 1048 | 117 |
| pdfplumber | 18.7 | 633 | 1054 | 150 |

Average word overlap: 97.8%  
Errors: 0

Decision: PyMuPDF selected as the primary parser. It was approximately 5.2x faster, clause counts were within 10%, and manual checks did not show a clear table-quality advantage for pdfplumber. No fallback enabled.

## Step 3: Clause-aware chunker

Chunks: 533 across 20 files (median words pending; 2 files failed processing). Manual review (10 chunks): 8/10 boundary_ok, 10/10 page_ok.  
Known limitations: tables flattened (flagged), footnotes, page range per clause.

## Step 4: Relationship ground truth

23 relations across 23 source documents (supersedes 12, amends 8, refers_to 2, withdraws 1).

Evidence: 23 pdf_text, 0 rbi_web.  
found_by: 23 manual, 0 finder. Status derived by rule from relations (not hand-labeled).

Relation validation: 0 errors, 0 warnings.  
Text-evidence relations present in 21/50 corpus documents.

## Step 5: Corpus (50 documents)

50 docs, 2564 chunks, corpus_sha=c6be8a2ed75e. Types: MD 20, AMD 12, CIR 18. Status: active 47, superseded 3.

Old+new pairs in corpus: 3; amendments with base in corpus: 4.

Ingestion errors: 0. Repeat ingestion: 50 cache hits, 0 new/rechunked documents.

Flagged for review: 14 (see eval/results/step5_ingest_report.csv). Detailed quality review deferred.

## Step 6: Golden dataset (dev split)

60 questions: simple 15, multihop 12 (6 graph-only), outdated_trap 12, what_changed 9, unanswerable 12.

Gold evidence = (file, page, quote), auto-verified against PDF text. 15 simple questions LLM-assisted drafts, all human-verified; 45 written entirely by me. Held-out test split (40 questions) will be added later and not used for tuning.

golden_sha=c2c4e0826b14, corpus_sha=c6be8a2ed75e.

## Step 7: Retrieval eval harness

Gold evidence: `(file, page, quote)`. Page-hit is the primary metric; quote-hit is strict. Evaluation includes 95% bootstrap confidence intervals and paired comparisons between runs.

| Run | R@5 | MRR@20 | quoteR@5 | Superseded in top-5 | Abstain AUC |
|---|---:|---:|---:|---:|---:|
| random | 2.1% | 0.01 | 0.0% | — | 0.500 |
| idf_overlap (throwaway lexical baseline) | 72.9% | 0.67 | 62.5% | 29.2% | 0.877 |
| oracle_page (ceiling, cheating) | 100.0% | 1.00 | 18.8% | — | 1.000 |

n = 48 answerable dev questions. Differences under ~3 percentage points are noise at this sample size.

95% bootstrap CIs are reported by the evaluation harness; the table above shows the observed metric values.

## Step 8: BM25 (from scratch) and naive dense retrieval (first ablation rows)

n = 48 answerable dev questions; 95% bootstrap CI; differences under ~3 percentage points are noise.

| Variant | R@5 [95% CI] | MRR@20 | quoteR@5 | Superseded in top-5 |
|---|---:|---:|---:|---:|
| idf_overlap (throwaway) | 72.9% | 0.67 | 62.5% | 29.2% |
| BM25 (title+heading context, stemmed) | 87.5% [81.2, 93.8] | 0.88 | 75.0% | 37.5% |
| BM25 text-only | 79.2% [69.8, 87.5] | 0.75 | 67.7% | 37.5% |
| BM25 no stemming | 84.4% [77.1, 91.7] | 0.81 | 68.8% | 39.6% |
| Dense bge-small (naive vector-only baseline) | 80.2% [70.8, 89.6] | 0.74 | 60.4% | 47.9% |
| Dense bge-small, no query prefix | 80.2% [68.8, 89.6] | 0.76 | 60.4% | 45.8% |

Embeddings are cached by SHA1 of passage text; only changed chunks are re-embedded.

Paired comparison: BM25 improved over IDF overlap on answerable R@5 (+14.6 percentage points; 95% CI [+5.2, +26.0]). BM25 and dense bge-small showed no clear difference in R@5; BM25 had higher MRR@20 and quoteR@5 in the paired comparison.
