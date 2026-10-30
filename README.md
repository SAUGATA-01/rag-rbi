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