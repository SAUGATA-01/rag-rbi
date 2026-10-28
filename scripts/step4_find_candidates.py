
"""Step 4: relation-sentence candidates from chunks (labeling HELPER, extraction nahi)."""
import argparse
import csv
import json
from pathlib import Path

from rbi_rag.relations import find_candidates

CHUNKS = Path("data/processed/chunks")
OUT = Path("data/processed/relation_candidates.csv")
STRONG = ("supersede", "repeal", "rescind", "withdraw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="", help="show candidates for files containing this text in the console")
    args = ap.parse_args()

    files = sorted(CHUNKS.glob("*.jsonl"))
    if not files:
        print("No JSONL files found in data/processed/chunks. Run Step 3 first.")
        return

    rows = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        rows += find_candidates(chunks)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "chunk_id", "page", "triggers", "sentence"])
        w.writeheader()
        w.writerows(rows)

    by_file = {}
    for r in rows:
        d = by_file.setdefault(r["file"], [0, 0])
        d[0] += 1
        d[1] += any(t in r["triggers"] for t in STRONG)
    print("file                                  candidates  strong(supersede/repeal/rescind/withdraw)")
    for fp in files:
        name = fp.stem + ".pdf"
        c, s = by_file.get(name, [0, 0])
        print(f"{name[:36]:<36} {c:>10} {s:>8}")
    strong_docs = sum(1 for c, s in by_file.values() if s)
    print(f"\nfiles with >=1 strong candidate: {strong_docs} / {len(files)}")
    print(f"CSV: {OUT}")

    if args.file:
        print(f"\n=== candidates for '{args.file}' ===")
        for r in rows:
            if args.file in r["file"]:
                print(f"\n[p{r['page']}] ({r['triggers']}) {r['chunk_id']}\n  {r['sentence']}")


if __name__ == "__main__":
    main()
