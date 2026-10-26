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