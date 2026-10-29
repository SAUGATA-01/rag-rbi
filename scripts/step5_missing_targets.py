
"""Step 5: Targets in relations.csv that are not in the corpus (download shopping list)."""
import csv
import re
from collections import defaultdict
from pathlib import Path

RELS = Path("data/metadata/relations.csv")
OUT = Path("data/processed/missing_targets.csv")


def main():
    if not RELS.exists():
        print("relations.csv not found (run Step 4).")
        return
    with RELS.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
    miss = [r for r in rows
            if not (r.get("dst_file") or "").strip()
            and r.get("relation") in ("supersedes", "amends", "withdraws")]
    groups = defaultdict(list)
    for r in miss:
        groups[re.sub(r"\s+", " ", r["dst_ref"]).strip().lower()].append(r)

    items = sorted(groups.values(), key=lambda g: -len(g))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dst_ref", "n_rows", "relations", "referenced_by", "pages"])
        for g in items:
            w.writerow([g[0]["dst_ref"], len(g), "|".join(sorted({r["relation"] for r in g})),
                        "|".join(sorted({r["src_file"] for r in g})),
                        "|".join(sorted({r["evidence_page"] for r in g}))])

    print(f"Targets outside the corpus: {len(items)} (relation rows: {len(miss)})\n")
    for g in items:
        rels = "/".join(sorted({r["relation"] for r in g}))
        srcs = ", ".join(sorted({r["src_file"] for r in g}))
        print(f"- {g[0]['dst_ref']}\n    {rels} <- {srcs}")
    print(f"\nCSV: {OUT}")
    print("Download these documents, name them using the MD_/AMD_/CIR_... convention, "
          "run make_manifest.py, then fill in dst_file in relations.csv.")


if __name__ == "__main__":
    main()
