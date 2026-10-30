# RBI-RAG

A from-scratch RAG system over RBI Master Directions and circulars.
Citation-graph aware, evaluation-first. Research project, not legal or compliance advice.

Status: Step 1 (setup and data feasibility test).

## Step 2: Parser comparison (20 RBI PDFs, up to 30 pages each)

| Parser | Text time (s) | Clause lines | Sub-clause lines | Tables found |
|---|---:|---:|---:|---:|
| PyMuPDF | 3.6 | 625 | 1048 | 117 |
| pdfplumber | 18.7 | 633 | 1054 | 150 |

Average word overlap: 97.8%
Errors: 0

Decision: PyMuPDF selected as the primary parser. It was approximately
5.2x faster, clause counts were within 10%, and manual checks did not
show a clear table-quality advantage for pdfplumber. No fallback enabled.

## Step 3: Clause-aware chunker

Cunks: 533 across 20 files (median words pending; 2 files failed processing). Manual review (10 chunks): 8/10 boundary_ok, 10/10 page_ok.
Known limitations: tables flattened (flagged), footnotes, page range per clause.

## Step 4: Relationship ground truth

20 relations across 20/20 documents (supersedes 11, amends 6, withdraws 1).

Evidence: 20 pdf_text, 0 rbi_web.

found_by: 20 manual, 0 finder. Status derived by rule from relations (not hand-labeled).

## Step 5: Corpus (50 documents)

50 docs, 2529 chunks, corpus_sha=ecd4be274dc8. Types: MD 20, AMD 12, CIR 18. Status: active 48, superseded 2.

Old+new pairs in corpus: 2; amendments with base in corpus: 3. Flagged for review: 14 (see eval/results/step5_ingest_report.csv).