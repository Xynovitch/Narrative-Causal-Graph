# Changelog

All notable changes to the Causal Event Knowledge Graph (CEKG) pipeline will be documented in this file.

## [Unreleased] - 2026-03-26

### Added
- **SAT Sentence Segmentation:** Replaced naive `split('. ')` with `wtpsplit` (`sat-3l` model) for proper sentence boundary detection during chunk splitting in `pipeline.py`. Falls back to regex when `wtpsplit` is not installed. Added `wtpsplit` to `requirements.txt`.
- **BM25 Keyword Candidate Pairs:** Added `get_bm25_pairs()` in `dynamic_context.py` using `rank_bm25.BM25Okapi`. For each event, retrieves the top-5 keyword-overlap matches across all events. Complements cosine similarity by catching named entities and domain terms that dense embeddings compress away. On *Great Expectations*, BM25 contributed **49% of all candidate pairs** (11,304 unique out of 23,203 total). Added `rank-bm25` to `requirements.txt`.
- **RAG Passage Context for Causal Assessment:** Added `cekg_pipeline/passage_index.py` (`PassageIndex` class). Segments the full novel into overlapping 400-char passages, embeds with `all-MiniLM-L6-v2`, and provides fast cosine-similarity retrieval. Top-2 passages are injected as `Narrative context` into each causal assessment LLM prompt to ground reasoning in actual text. Index built from chapter texts after `text_split` stage; passed through `_causal_linking` to `process_pairs_causal_only` and `assess_pairs_causal`.
- **Orphan Scene Fallback:** After LLM scene generation, events not assigned to any scene are grouped into per-chapter catch-all scenes. Eliminates the 604 orphan events seen in the previous run.
- **Stale CSV Cleanup:** `export_csv` in `exporters.py` now deletes any `.csv` files in `out_dir` that are not part of the current export set before writing. Removes leftover files from schema changes (e.g. `semantic_links.csv`, `cosmograph_edges.csv`).
- **Run Report:** `run_report_2026-03-26.md` documenting metrics, feature validation, known issues, and improvement areas.

### Changed
- **Candidate pair threshold lowered:** Default `thematic_threshold` changed from `0.95` → `0.50` in `dynamic_context.py`, `pipeline.py`, and `main.py`. Previous 0.95 threshold produced only 12 long-shot pairs; 0.50 produces 2,838. Total candidate pairs: 7,272 → 23,203 (+219%).
- **`process_pairs_causal_only` signature:** Added `passage_index` parameter (passed from pipeline to causal linker).
- **`assess_pairs_causal` signature:** Added `passage_index` parameter; activates RAG context injection when index is ready.

### Removed
- **`ThematicLink` edges:** Removed `ThematicLink` dataclass from `schemas.py`, `build_thematic_links()` from `theme_annotation.py`, and all THEMATIC edge exports from `exporters.py` and `graph_mapper.py`. Per the original CEKG spec, theme data lives as node properties (`theme_annotations` on `CEKEvent`), not as pairwise edges. The previous implementation generated ~3M edges which were architecturally incorrect.

### Fixed
- **Rate-limit retry bypass (chapter 59):** `_async_llm_json_call` was swallowing all exceptions and returning `[]` after 3 short (1–2s) retries. This prevented `_process_chunk_with_retry`'s rate-limit-aware 30–90s backoff from ever activating. Fix: rate-limit errors (HTTP 429, "rate limit", "quota") now re-raise immediately, propagating up to the outer retry loop where the proper backoff is applied.
- **Infinite loop in `PassageIndex._segment_text`:** When `end - overlap <= start` (near end of text), `next_start` could fail to advance. Added forward-progress guarantee: `if next_start <= start: next_start = start + passage_size`.
- **`_process_chapter_chunked` concurrency:** Added `asyncio.Semaphore(3)` per chapter to cap concurrent chunk API calls. Prevents burst rate-limit triggers when a chapter has 10+ chunks.
- **Chunk retry logic:** Increased `max_retries` from 3 → 5 in `_process_chunk_with_retry`. Added rate-limit detection with 30–90s progressive backoff vs. exponential backoff for other errors.

## [Unreleased] - 2025-11-17

### Fixed
- **Windows Compatibility:** Resolved `UnicodeEncodeError` on Windows terminals by removing special characters (✓, →) from console output logs in `main.py`.
- **Prompt Formatting:** Fixed `KeyError: 'name'` crashes caused by f-string conflicts in `llm_service.py`. Escaped JSON example braces (`{{ }}`) to prevent premature formatting.
- **Graph Validation Logging:** Corrected logic in `pipeline.py` where "No Relation" responses were silently skipped. Added a `no_relation_edges` counter to distinguish between valid rejections (cycles) and non-causal pairs.
- **Model Compatibility (O1/GPT-5):** Fixed `BadRequestError: 400` when using reasoning models.
    - Added automatic detection for `o1` and `gpt-5` models.
    - Switched API parameters to use `max_completion_tokens` instead of `max_tokens`.
    - Disabled `temperature`, `logprobs`, and `response_format` for reasoning models to comply with API strictness.
- **JSON Extraction:** Resolved `ExtractionError: No JSON found` for "chatty" reasoning models.
    - Implemented robust Regex extraction to locate JSON within Markdown code blocks (` ```json ... ``` `).
    - Added fallback logic to repair truncated JSON responses caused by token limits.
- **Neo4j Export:** Fixed `Neo.ClientError` syntax errors during Cypher import.
    - Added strict escaping to Node IDs and Relationship IDs in `exporters.py` to handle IDs containing double quotes (e.g., `whyfactor_"he"_said`).
- **Agent Semantics:** Fixed nonsensical actor extraction (e.g., "Narrator's liver"). Updated prompts in `llm_service.py` to explicitly constrain Actors and Patients to sentient beings only.

### Changed
- **Token Limits:** Increased default token limit to 16,000 for `extract_events_from_text` to accommodate larger batch sizes (5 paragraphs/call).
- **Logging:** Improved error logging in `llm_service.py` to print the first 200 characters of raw output when parsing fails, aiding in debugging.

### Added
- **Sanitization:** Added `_sanitize_name_for_id` utility in `graph_mapper.py` to strip quotes and special characters from canonical entity IDs before graph generation.

Here is the changelog entry for the **new features** and refactoring work we completed in this session. You can add this to the top of your `CHANGELOG.md` file.

## [Unreleased] - feature/dynamic-ontology-refactor

### Major Architecture Changes
- **Dynamic Ontology Injection:** The pipeline no longer relies on hardcoded Enums for `EventType` or `RelationType`. Both are now injected dynamically into LLM prompts from external JSON files (`event_ontology.json`, `relationship_ontology.json`), allowing for flexible, domain-specific taxonomies.
- **Node-to-Attribute Refactor:**
    - **Removed:** Explicit `Place` and `Time` nodes.
    - **Changed:** `Location` and `Time` are now first-class attributes (`location_context`, `time_context`) on the `Event` node itself.
    - **Reasoning:** This simplifies the graph structure, reduces node explosion, and allows for easier "Event-centric" querying (e.g., "Find all events in the Kitchen").

### Added
- **`generate_ontology.py`:** A new script to bootstrap an **Event Type Dictionary** by analyzing raw text from a directory of novels (Bottom-up extraction) and optionally merging with a Narrative Theory seed (Top-down).
- **`generate_relationship_ontology.py`:** A companion script to extract and standardize **Relationship Types** (e.g., `EMOTIONAL_TRIGGER`, `SOCIAL_OBLIGATION`) from text.
- **`propagate_context_attributes` (in `graph_builder.py`):** A new pass that propagates `location_context` and `time_context` linearly through the narrative chain to fill in implicit blanks before graph generation.
- **Semantic Linking:** Fully integrated `SemanticLink` (non-causal edges like `EXPLAINS`, `CONTRASTS`) into the extraction and export pipeline.

### Changed
- **Schema (`schemas.py`):**
    - `CEKEvent` now separates `raw_description` (natural language text) from `event_category` (standardized ontology label).
    - `RelationType` is now a `str` (was `Enum`) to support dynamic dictionaries.
- **LLM Service (`llm_service.py`):**
    - Prompts updated to request the new JSON structure (`raw_description`, `location_context`, etc.).
    - `batch_extract_events` and `batch_assess_pairs` now accept `event_ontology` and `relationship_ontology` arguments.
- **Pipeline (`pipeline.py`):**
    - `run_async` now loads ontology JSON files at startup and passes them to the extraction service.
    - Restored progress logging (`print`) for batch/chunk processing.
- **Exporters (`exporters.py`):**
    - `export_neo4j_cypher` completely rewritten to use `UNWIND` batches for faster import and to handle the new property-based schema (Events have properties, not links to Place nodes).
    - `export_csv` updated to export the new event attributes and stop exporting `places.csv`.

### Fixed
- **Pipeline Crashes:** Resolved `AttributeError: module 'schemas' has no attribute 'SemanticLink'` by ensuring all dataclasses are correctly defined.
- **Propagate Logic:** Fixed `AttributeError` in `graph_builder` by implementing the missing `propagate_context_attributes` function.
- **LLM Output Handling:** Fixed `AttributeError: 'dict' object has no attribute 'strip'` in ontology generators by adding sanitization logic for LLM responses that return objects instead of strings.