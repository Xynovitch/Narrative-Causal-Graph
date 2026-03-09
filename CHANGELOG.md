# Changelog

All notable changes to the Causal Event Knowledge Graph (CEKG) pipeline will be documented in this file.

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