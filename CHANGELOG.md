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