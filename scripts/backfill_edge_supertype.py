"""Fill missing edge_supertype on CausalEdge entries in an exported JSON-LD file.

The dataset at docs/web/data/great_expectations_gpt5.json was produced before
FINE_TO_SUPERTYPE in cekg_pipeline/theme_annotation.py covered gpt-5's relation
vocabulary, so ~9k causal edges have edge_supertype: null. The map has since
been backfilled (commit 90a1396); this script applies that map to the on-disk
file so we don't have to lean on the JS-side fallback.

Usage:
    python scripts/backfill_edge_supertype.py docs/web/data/great_expectations_gpt5.json
    python scripts/backfill_edge_supertype.py path/to/file.json --dry-run
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Make `cekg_pipeline` importable when running from anywhere in the repo.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cekg_pipeline.theme_annotation import FINE_TO_SUPERTYPE


def backfill(path: Path, dry_run: bool = False) -> int:
    data = json.loads(path.read_text())
    graph = data.get("@graph", [])

    filled = 0
    still_missing: Counter[str] = Counter()
    already_set = 0
    causal_total = 0

    for item in graph:
        if item.get("type") != "CausalEdge":
            continue
        causal_total += 1
        if item.get("edge_supertype"):
            already_set += 1
            continue
        rt = (item.get("relationType") or "").strip()
        st = FINE_TO_SUPERTYPE.get(rt) or FINE_TO_SUPERTYPE.get(rt.upper())
        if st:
            item["edge_supertype"] = st
            filled += 1
        else:
            still_missing[rt or "<empty>"] += 1

    print(f"causal edges: {causal_total}")
    print(f"  already had supertype: {already_set}")
    print(f"  filled by this script: {filled}")
    print(f"  still missing:         {sum(still_missing.values())}")
    if still_missing:
        print("\nrelation types not in FINE_TO_SUPERTYPE (top 20):")
        for rt, n in still_missing.most_common(20):
            print(f"  {n:>6}  {rt}")

    if filled and not dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"\nwrote {path}")
    elif dry_run:
        print("\n(dry run — no file written)")
    else:
        print("\nnothing to backfill")
    return filled


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="JSON-LD file to patch in place")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    backfill(args.path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
