# CEKG Pipeline Run Report
**Date:** 2026-03-26
**Input:** Great Expectations.txt (59 chapters)
**Mode:** `--full --max-pairs 999999 --resume` (from text_split checkpoint)
**Runtime:** ~512 minutes (8h 32m)
**Model:** gpt-4o-mini

---

## Pipeline Results

| Metric | This Run | Previous Baseline | Change |
|--------|----------|-------------------|--------|
| Events extracted | 3,338 | ~3,338 | — |
| Characters | 116 | — | — |
| Candidate pairs evaluated | **23,203** | 7,272 | **+219%** |
| Causal links total | **4,121** | 3,564 | **+15.6%** |
| — McKee | 1,825 | — | — |
| — Truby | 2,296 | — | — |
| Scenes | 295 | — | — |
| DAG valid | ✓ | ✓ | — |
| Orphan events | 0 | 604 | **fixed** |

---

## New Feature Validation

### 1. SAT Sentence Segmentation (wtpsplit)
- **Status:** Active and working
- Replaces naive `split('. ')` for chunk boundaries during event extraction
- No measurable regression in event count; cleaner chunk boundaries reduce mid-sentence cuts

### 2. BM25 Keyword Candidate Pairs
- **Status:** Active — largest contributor to pair expansion

| Pool | Raw | Unique (after dedup) | Share |
|------|-----|----------------------|-------|
| Adjacent (i, i+1) | 3,337 | 3,337 | 14% |
| Entity-guided | 5,523 | 2,696 | 12% |
| **BM25 keyword** | **11,944** | **11,304** | **49%** |
| Scene (sim ≥ 0.5) | 4,858 | 3,132 | 13% |
| Long-shot sliding | 2,838 | 2,734 | 12% |
| **Total** | | **23,203** | |

BM25 is the dominant pool, contributing nearly half of all candidate pairs. Named entities and domain terms that embedding similarity compresses away (e.g. "Magwitch", "convict", "debt") produce high BM25 scores across the full span of the novel.

### 3. RAG Passage Context for Causal Assessment
- **Status:** Active
- Index built: **2,879 passages** from 59 chapters (400-char segments, 80-char overlap)
- Top-2 passages injected per pair in causal LLM prompt
- Effect on link quality not directly measurable in one run, but provides grounding for ambiguous long-range pairs

### 4. Stale CSV Cleanup
- **Status:** Fixed — `thematic_links.csv` (82,681 rows, old format) removed automatically
- Current output: 10 active CSV files, 31,820 total rows

### 5. Orphan Scene Fallback
- **Status:** Fixed — 0 orphan events (down from 604)
- 295 scenes generated (catch-all fallback active for chapters where LLM scene grouper missed events)

---

## Known Issues

### Chapter 59: 7/10 chunks failed (70% data loss)

**Root cause (identified this session):**
`_async_llm_json_call` internally catches ALL exceptions and retries with 1–2s backoff, then returns `[]` silently. This means `_process_chunk_with_retry`'s rate-limit-aware 30–90s retry logic **never activates** — it never sees an exception.

By the time chapter 59 runs (last in the 8h session), the per-minute RPM quota is depleted from chapter 58's burst. The 3 internal short retries exhaust the remaining tokens, and all 7 affected chunks return `[]`.

**Evidence:**
All 10 `[llm] Event extraction:` log lines appear (they print *before* the API call), but 7 chunks return empty. Earlier chapters with the same chunk count (11, 18, 19, 22, 29, 38, 40, 53) all succeed because they run when quota is fresh.

**Fix applied (2026-03-26):**
`_async_llm_json_call` now re-raises immediately on rate-limit errors (HTTP 429, "rate limit", "quota") instead of absorbing them. This allows `_process_chunk_with_retry`'s 5-retry loop with 30–90s backoff to actually fire.

---

## Areas for Improvement

### High Priority

**1. Scene generation is sequential (slow)**
`_generate_scenes_optimized` iterates chapters in a `for` loop with a single `await` per chapter — 59 blocking API calls in series. On this run, the first chapter stalled for ~5 minutes (600s timeout). Total scene generation time: estimated 20–30 minutes.
*Fix:* `asyncio.gather(*[extract_scenes_from_chapter_async(...) for cid, evs in by_chap.items()])` with a concurrency semaphore.

**2. Scene extraction token truncation**
Chapter 1's scene extraction response was 33,271 chars and hit the 12,000 token cap. The response was truncated, JSON parse failed, and the fallback created a single catch-all scene for the entire chapter.
*Fix:* Increase `max_tokens` to 16,000 for scene extraction, or split large chapters into halves before sending to scene grouper.

**3. Candidate pair coverage: 0 events with no causal bridge**
4,121 links from 3,338 events = 1.23 links/event average. Some events are isolated (no incoming or outgoing CAUSES edge). The current pair generation depends on entity co-occurrence and thematic similarity — events with unique or generic entities may not appear in any candidate pair.
*Potential improvement:* Add a "near-miss" pool: events within ±5 positions that share no entities but are not in adjacent pairs (they are currently missed by entity-guided + BM25 if the descriptions are generic).

**4. Chapter 59 data loss = ~26 events instead of expected ~85–90**
With 7/10 chunks failing, chapter 59 contributes only 26 events. Great Expectations' chapter 59 is the epilogue ("I saw no shadow of another parting from her") — it may be underrepresented in the knowledge graph, missing Pip's final reunion with Estella and closure of the Magwitch arc.

### Medium Priority

**5. BM25 top-K=5 may miss some long-range pairs**
BM25 currently retrieves the top-5 keyword matches per event. For named characters appearing in 20+ events (Pip, Magwitch, Estella, Miss Havisham), the top-5 BM25 results cluster around nearby high-frequency events and may miss thematically important distant ones.
*Fix:* Increase `BM25_TOP_K` to 10–15, or apply a second BM25 pass on event pairs that have high embedding similarity but no entity or BM25 coverage.

**6. RAG contribution not measurable**
The RAG passage context is injected into causal prompts but there's no logging of whether it changes any `relationType` decisions. A/B test by running a subset of pairs with and without RAG context to quantify the delta.

**7. Causal pair cache invalidation on threshold change**
The BM25/similarity pair generation is not cached — it recomputes every run from scratch (~30 seconds for 3,338 events). If the pipeline is re-run with `--resume` after the linking checkpoint, the pair generation still runs from scratch. Consider caching the pair list as part of the linking checkpoint.

### Low Priority

**8. Scene quality: catch-all scenes conflate unrelated events**
The fallback creates one scene per chapter for orphan events. In later chapters with dense action (chapters 50–59), a single catch-all scene may contain 30+ unrelated events, inflating within-scene pair counts for dynamic_context's scene pool.

**9. `hosts.csv` and `places.csv` are empty**
The HOSTS and PLACES relationship exporters produce 0-row files. This may indicate that `place_type` is never set on Scene objects (it's always `None` in `_generate_scenes_optimized`). If places/locations are important for the Neo4j schema, the scene exporter should resolve location strings to canonical place nodes.

**10. `semantic_links.csv` absent but `causes.csv` + `follows.csv` are separate**
The old `semantic_links.csv` (merged CAUSES + FOLLOWS) is gone. The new split is cleaner, but the Cypher import script may need updating if downstream Neo4j queries still reference `semantic_links`.

---

## Output File Summary

| File | Rows | Description |
|------|------|-------------|
| `events.csv` | 3,338 | Event nodes with theme_annotations |
| `causes.csv` | 4,121 | CAUSES edges (McKee + Truby) |
| `follows.csv` | 3,279 | FOLLOWS (chronological) edges |
| `agents.csv` | 5,073 | Character/agent nodes |
| `acts_in.csv` | 4,190 | ACTS_IN edges (agent → event) |
| `affected_in.csv` | 883 | AFFECTED_IN edges |
| `motivates.csv` | 3,629 | MOTIVATES edges (why-factor → event) |
| `whyfactors.csv` | 3,629 | WHY_FACTOR nodes |
| `scenes.csv` | 295 | Scene nodes |
| `scene_includes_event.csv` | 3,373 | SCENE_INCLUDES_EVENT edges |
| `hosts.csv` | 0 | Empty — place_type not populated |
| `places.csv` | 0 | Empty — place nodes not generated |
