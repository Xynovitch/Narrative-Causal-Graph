"""
One-shot patch for chapter 59 of the gpt-5 full run.

Root cause: the chapter splitter absorbed the Project Gutenberg license
footer into chapter 59. The novel actually ends at the *** END OF THE
PROJECT GUTENBERG EBOOK *** marker (~7,650 chars in); the remaining
~18,500 chars are pure license text. That triggered (a) deterministic
gpt-5 chunk failures and (b) hallucinated events from license verbiage.

This patch trims chapter 59 in the text_split checkpoint, re-extracts
from clean narrative, splices into the extraction checkpoint, and clears
downstream stage checkpoints so `--resume` re-derives them.

Run from project root:  python scripts/patch_chapter_59.py
"""
import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import asdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cekg_pipeline.pipeline import CEKGPreprocessor
from cekg_pipeline.checkpoint_manager import CheckpointManager
from cekg_pipeline.text_processor import strip_gutenberg_boilerplate

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_gpt5_full")
RUN_ID = "Great Expectations.txt_1762146485"
TARGET_CHAPTER = 59


async def main():
    mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR, run_id=RUN_ID)

    ext = mgr.load_checkpoint("extraction")
    txt = mgr.load_checkpoint("text_split")
    if ext is None or txt is None:
        print("[patch] FAIL: missing extraction or text_split checkpoint")
        return 1

    chapters = txt["chapters"]
    try:
        ch59_idx = next(i for i, (cid, _) in enumerate(chapters) if cid == TARGET_CHAPTER)
    except StopIteration:
        print(f"[patch] FAIL: chapter {TARGET_CHAPTER} not in text_split")
        return 1
    ch59_text_orig = chapters[ch59_idx][1]

    # Trim Project Gutenberg license boilerplate from the end of chapter 59
    ch59_text = strip_gutenberg_boilerplate(ch59_text_orig)
    print(f"[patch] Chapter {TARGET_CHAPTER}: {len(ch59_text_orig)} chars -> "
          f"{len(ch59_text)} chars after PG trim "
          f"({len(ch59_text_orig) - len(ch59_text)} chars removed)")

    # Persist the trimmed chapter back to text_split so future resumes
    # see the corrected text instead of the license-padded version.
    chapters[ch59_idx] = (TARGET_CHAPTER, ch59_text)
    mgr.save_checkpoint(
        "text_split",
        {"chapters": chapters, "max_chapters": txt.get("max_chapters")},
        description=f"Patched: trimmed PG license from chapter {TARGET_CHAPTER}",
    )

    old_events = ext["events"]
    old_produces = ext["produces"]
    old_occ = ext["entity_occurrences"]

    ch59_old_ids = {e["id"] for e in old_events if e["chapter"] == TARGET_CHAPTER}
    print(f"[patch] Old chapter {TARGET_CHAPTER}: {len(ch59_old_ids)} events to discard")

    kept_events = [e for e in old_events if e["chapter"] != TARGET_CHAPTER]
    kept_produces = [p for p in old_produces if p["event_id"] not in ch59_old_ids]
    kept_occ = {}
    for k, v in old_occ.items():
        filtered = [(eid, seq) for eid, seq in v if eid not in ch59_old_ids]
        if filtered:
            kept_occ[k] = filtered

    print(f"[patch] Kept from chapters 1-58: {len(kept_events)} events, "
          f"{len(kept_produces)} produces, {len(kept_occ)} entity buckets")

    pre = CEKGPreprocessor(
        openai_model="gpt-5",
        schema_path=os.path.join(PROJECT_ROOT, "schema.json"),
        checkpoint_dir=CHECKPOINT_DIR,
        enable_checkpoints=False,
    )

    # Pre-warm the resolver with every canonical actor/patient name learned
    # across chapters 1-58. Otherwise chapter 59 extraction would start with
    # only the seeded aliases and might canonicalize "Joe" or "Magwitch" to a
    # fresh entity, fragmenting characters in the final graph.
    canonical_names = set()
    for p in kept_produces:
        if p["entity_type"] in ("actor", "patient"):
            canonical_names.add(p["entity_name"])
    for name in canonical_names:
        pre.resolver.register_character(name)
    print(f"[patch] Pre-registered {len(canonical_names)} canonical character names")

    # Continue the global sequence numbering from where chapters 1-58 left off
    pre.global_event_sequence = len(kept_events)

    print(f"\n[patch] Re-extracting chapter {TARGET_CHAPTER}...")
    new_events, new_produces, new_occ = await pre._process_chapter_chunked(
        ch59_text,
        TARGET_CHAPTER,
        enable_confidence_calibration=True,  # matches --full
        extraction_style="detailed",          # matches pipeline.py:785
        chunk_size=3000,                      # matches default
    )
    print(f"\n[patch] New extraction: {len(new_events)} events, "
          f"{len(new_produces)} produces, {len(new_occ)} entity buckets")

    # With the PG license trimmed, chapter 59 is short pure narrative — we
    # expect ~30-60 events from ~3 chunks. Anything zero is a hard fail;
    # anything below 10 is suspicious enough to surface but not auto-abort.
    if len(new_events) == 0:
        print("[patch] ABORT: re-extraction returned zero events. Extraction checkpoint untouched.")
        return 2
    if len(new_events) < 10:
        print(f"[patch] WARNING: re-extraction returned only {len(new_events)} events. "
              f"Continuing, but inspect output before resuming.")

    final_events = kept_events + [asdict(e) for e in new_events]
    final_produces = kept_produces + [asdict(p) for p in new_produces]
    final_occ = defaultdict(list)
    for k, v in kept_occ.items():
        final_occ[k] = list(v)
    for k, v in new_occ.items():
        final_occ[k].extend(v)

    print(f"[patch] Spliced totals: {len(final_events)} events, "
          f"{len(final_produces)} produces, {len(final_occ)} entity buckets")

    ok = mgr.save_checkpoint(
        "extraction",
        {
            "events": final_events,
            "produces": final_produces,
            "entity_occurrences": dict(final_occ),
            "global_event_sequence": pre.global_event_sequence,
        },
        description=f"Patched chapter {TARGET_CHAPTER}; total {len(final_events)} events",
    )
    if not ok:
        print("[patch] FAIL: save_checkpoint returned False")
        return 3

    for stage in ("context_propagation", "agent_classification"):
        if mgr.has_checkpoint(stage):
            mgr.clear_checkpoint(stage)

    print("\n[patch] Done. Resume the pipeline with:")
    print("  python main.py --input 'Great Expectations.txt' --full --openai-model gpt-5 \\")
    print("    --max-concurrent-calls 4 --checkpoint-dir ./checkpoints_gpt5_full \\")
    print("    --out-json ./gpt5_full.json --out-cypher ./gpt5_full_import.cypher \\")
    print("    --out-csv ./neo4j_csv_gpt5_full --resume")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
