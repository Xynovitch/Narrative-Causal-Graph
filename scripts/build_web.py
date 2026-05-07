#!/usr/bin/env python3
"""
Copy a pipeline JSON-LD output (e.g. ./gpt5_full.json) into docs/web/data/
and update docs/web/data/manifest.json so the webpage can list it.

Usage:
    python scripts/build_web.py <source.json> <novel-key> "<Novel Label>"

Example:
    python scripts/build_web.py gpt5_full.json great_expectations "Great Expectations (gpt-5)"
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = ROOT / "docs" / "web" / "data"


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    src = Path(sys.argv[1]).resolve()
    novel_key = sys.argv[2]
    novel_label = sys.argv[3]

    if not src.exists():
        print(f"[error] source file not found: {src}")
        sys.exit(1)

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    dest_filename = f"{novel_key}.json"
    dest = WEB_DATA / dest_filename

    print(f"[build] copying {src} -> {dest}")
    shutil.copy2(src, dest)

    # Quick stats for the manifest entry
    with open(dest) as f:
        graph = json.load(f).get("@graph", [])
    counts = {}
    for item in graph:
        t = item.get("type", "?")
        counts[t] = counts.get(t, 0) + 1
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"[build] {dest_filename}: {size_mb:.1f} MB, types={counts}")

    manifest_path = WEB_DATA / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"novels": [], "default": None}

    novels = [n for n in manifest.get("novels", []) if n.get("key") != novel_key]
    novels.append({
        "key": novel_key,
        "label": novel_label,
        "file": dest_filename,
        "events": counts.get("Event", 0),
        "causal_edges": counts.get("CausalEdge", 0),
        "thematic_edges": counts.get("ThematicEdge", 0),
        "size_mb": round(size_mb, 2),
    })
    novels.sort(key=lambda n: n["key"])
    manifest["novels"] = novels
    if not manifest.get("default"):
        manifest["default"] = novel_key

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build] updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
