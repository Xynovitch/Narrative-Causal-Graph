# CEKG Developer Manual

**Causal Event Knowledge Graph (CEKG)** — Developer Manual for developers with no prior knowledge of this codebase. This document is based strictly on the code present in the repository.

---

## 1. Project Architecture & Pipeline Flow

### 1.1 High-Level Summary

The CEKG project is a **Python pipeline** that:

1. **Ingests raw narrative text** (e.g., novels) and splits it into chapters.
2. **Extracts narrative events** (who did what, where, when, why) using an LLM (OpenAI), with optional ontology-based event types and coreference resolution.
3. **Propagates context** (location, time, actors, motivations) along the event sequence and builds **entity–event** links.
4. **Optionally classifies agent types** (e.g., protagonist, antagonist) and **groups events into scenes**.
5. **Finds causal and semantic links** between events (McKee/Truby theory-based relations and thematic links), with smart candidate-pair filtering to reduce cost.
6. **Exports** the resulting graph as **JSON-LD**, **Neo4j Cypher**, and **CSV** for Neo4j import.

The pipeline supports **checkpointing**: each major stage can be saved and resumed (e.g., after interruption or for debugging).

### 1.2 Pipeline Flow (Data Steps)

```
Raw text file (.txt)
    │
    ▼
[text_processor] load_text() → split_chapters()
    │
    ▼
Chapters: List[tuple[int, str]]  (chapter_id, chapter_text)
    │
    ▼
[pipeline] _process_chapter_chunked() → [llm_service] extract_events_from_text()
    │
    ▼
Per-chunk: (event_data_list, logprobs)
    │
    ▼
[pipeline] _parse_event_json_data() + coreference_resolver
    │
    ▼
all_events, all_produces, entity_occurrences  (in-memory)
    │
    ▼
[graph_builder] propagate_context_attributes() → propagate_context() → create_entity_to_event_links()
    │
    ▼
all_events (updated), all_produces (extended), entity_to_event_links
    │
    ▼
(Optional) [pipeline] _classify_agent_types()  →  agent_classifications
(Optional) [pipeline] _generate_scenes_optimized()  →  scenes
    │
    ▼
[pipeline] _integrated_causal_and_semantic_linking()
    │  uses [optimized_linking] IntelligentCausalLinker.get_candidate_pairs()
    │  uses [integrated_semantic] process_pairs_with_semantic_linking() or bulk causal-only
    ▼
causal_links, semantic_links
    │
    ▼
[exporters] build_jsonld() → export_json()  →  ge_preprocessed.json
[exporters] export_csv()                    →  neo4j_csv/*.csv
[graph_mapper] map_to_generic_graph()       →  GenericNode, GenericRelationship
[exporters] export_neo4j_cypher()           →  ge_import.txt (Cypher script)
```

**Checkpoints** are saved after: `text_split`, `extraction`, `context_propagation`, `agent_classification` (if enabled), `linking`, and `scenes` (if enabled). They are stored under `./checkpoints/<run_id>/` as pickle + JSON.

### 1.3 Entry Point

- **Entry point script:** `main.py`
- **What it does:** Parses CLI arguments, sets feature flags (from `--fast` / `--full` or individual flags), validates `OPENAI_API_KEY`, optionally clears or lists checkpoints, then instantiates `CEKGPreprocessor` (from `cekg_pipeline.pipeline`) and runs `asyncio.run(preprocessor.run_async(...))`.
- **Where the pipeline lives:** All orchestration and stage logic is in `cekg_pipeline/pipeline.py` inside the class `CEKGPreprocessor` and its async method `run_async()`.

---

## 2. Core Data Structures

All core data structures are defined in **`cekg_pipeline/schemas.py`**. They are dataclasses used to pass data between pipeline stages and into exporters.

### 2.1 Domain / Graph Entities

| Dataclass | Purpose | Key Fields |
|-----------|----------|-------------|
| **CEKEvent** | Central node: one narrative event. | `id`, `raw_description`, `action_type`, `time_context`, `location_context`, `actors`, `patients`, `chapter`, `sequence`, `confidence`, `source_quote`, `why_factors`, `theory` (@McKee / @Truby). |
| **EventProducesEntity** | Edge: Event → Entity (event “produces” an actor/patient/whyfactor/place/time). | `event_id`, `entity_id`, `entity_name`, `entity_type`, `relationship` (e.g. PRODUCES_ACTOR), `strength`, `agent_type`, `theory`. |
| **EntityPointsToEvent** | Edge: Entity → Event (entity participates in / motivates / hosts event). | `entity_id`, `entity_name`, `entity_type`, `next_event_id`, `relationship` (ACTS_IN, AFFECTED_IN, MOTIVATES, HOSTS), `strength`. |
| **CausalLink** | Edge: Event → Event (causal relation). | `source_event_id`, `target_event_id`, `relation_type`, `mechanism`, `weight`, `confidence`, `theory`, `directionality` (uni/bi). |
| **SemanticLink** | Edge: Event(s) → Event(s) (non-causal semantic relation). | `id`, `source_event_ids`, `target_event_ids`, `relation`, `cue`, `confidence`. |
| **Scene** | Grouping of events (scene-centric structure). | `id`, `chapter`, `included_event_ids`, `primary_location`, `time_period`, `participants`, `theme`, `summary`, `confidence`, `place_type`, `time_type`; extended: `all_actors`, `all_patients`, `all_whyfactors`. |

**Why these matter for the knowledge graph:** Events are the primary nodes; entities (actors, patients, why-factors) are linked to events via PRODUCES and ACTS_IN/AFFECTED_IN/MOTIVATES/HOSTS. Causal and semantic links form the event–event graph. Scenes group events and tie entities to scenes for scene-centric export.

### 2.2 Ontology / Schema (from ontology_loader)

Defined in **`cekg_pipeline/ontology_loader.py`** (used by pipeline, not in schemas.py):

| Dataclass | Purpose |
|-----------|---------|
| **EventType** | One event type from schema: `name`, `theory`, `description`, optional Neo4j metadata. |
| **RelationType** | One causal relation type: `name`, `theory`, `directionality`, `description`, optional Neo4j metadata. |
| **AgentType** | One agent type: `name`, `theory`, `description`, optional Neo4j metadata. |

### 2.3 Generic Graph Export

| Dataclass | Purpose |
|-----------|---------|
| **GenericNode** | Generic node for any graph DB: `uid`, `label`, `properties` (dict). |
| **GenericRelationship** | Generic edge: `start_node_uid`, `end_node_uid`, `rel_type`, `properties`. |

Used by `graph_mapper.map_to_generic_graph()` and then by `exporters.export_neo4j_cypher()`.

### 2.4 Other Structures

- **AgentRole** (schemas.py): Agent role in narrative; maps to AgentTypeDictionary.
- **PlaceContext**, **TimeContext** (schemas.py): Place/time with narrative role; mapped from schema PlaceTypeDictionary / TimeTypeDictionary (referenced in schema; used for typing/scene metadata).
- **EventPair** (optimized_linking.py): Lightweight pair for linking: `cause_id`, `effect_id`, `cause_text`, `effect_text`, `cause_seq`, `effect_seq`.

### 2.5 Exceptions (schemas.py)

- **CEKGError** — base.
- **ExtractionError** — event extraction failed.
- **DAGViolationError** — graph contains cycles.

---

## 3. Component Documentation (Script by Script)

### 3.1 `main.py`

- **Responsibility:** CLI entry point; argument parsing; preset application (--fast / --full); checkpoint clear/list; creation of `CEKGPreprocessor` and single call to `preprocessor.run_async(...)`.
- **Key:** `main()` → `asyncio.run(preprocessor.run_async(...))`. Reads `OPENAI_API_KEY` and `OPENAI_MODEL` from `cekg_pipeline.config`.
- **Dependencies:** `cekg_pipeline.pipeline` (CEKGPreprocessor), `cekg_pipeline.checkpoint_manager` (CheckpointManager), `cekg_pipeline.config` (OPENAI_API_KEY, OPENAI_MODEL).

---

### 3.2 `cekg_pipeline/config.py`

- **Responsibility:** Loads `.env` via `python-dotenv`; exposes `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`), `BATCH_SIZE`, `CAUSAL_BATCH_SIZE`, `SAMPLE_RATE`, `CACHE_MAX_SIZE`; defines `CONTROLLED_ACTION_ONTOLOGY` (verb → category map).
- **Dependencies:** `os`, `dotenv`.

---

### 3.3 `cekg_pipeline/schemas.py`

- **Responsibility:** Defines all domain dataclasses and generic graph dataclasses listed in Section 2, plus custom exceptions.
- **Key:** No logic; pure data definitions. Used by pipeline, graph_builder, graph_mapper, exporters, ontology_loader (indirectly via type expectations).
- **Dependencies:** `dataclasses`, `typing`.

---

### 3.4 `cekg_pipeline/text_processor.py`

- **Responsibility:** Load raw text and split it into chapters or paragraphs.
- **Key functions:**
  - `load_text(path: str) -> str` — read file as UTF-8.
  - `split_chapters(text: str) -> List[tuple[int, str]]` — split by chapter markers (regex for "Chapter N", "CHAPTER N", Roman numerals, etc.); if none, fall back to paragraph enumeration.
  - `split_into_paragraphs(text: str) -> List[str]` — split into paragraphs, with long paragraphs sub-split by sentences.
- **Dependencies:** `re`, `typing`.

---

### 3.5 `cekg_pipeline/llm_service.py`

- **Responsibility:** All OpenAI calls: event extraction, bulk causal assessment, agent classification, scene grouping; caches and retries with dynamic `max_tokens` to reduce truncation.
- **Key functions:**
  - `init_openai_client(api_key: str)` — returns OpenAI client.
  - `_async_llm_json_call(prompt, model, client, cache, cache_key, max_tokens)` — single async LLM call with cache; infers operation type and sets token limits; returns `(data, logprobs)`.
  - `extract_events_from_text(text_input, chapter_id, model, client, ...)` — returns `(events_list, logprobs)`.
  - `assess_pairs_bulk(pairs_batch, model, client, relation_ontology)` — returns list of causal assessment results per pair.
  - `classify_agent_type(character_name, event_descriptions, agent_type_names, model, client)` — returns agent type string.
  - `extract_scenes_from_chapter_async(chapter_events, chapter_id, model, client)` — returns list of scene dicts.
  - `get_cache_sizes()` — returns dict of cache sizes.
- **Caches:** `event_extraction_cache`, `assessment_cache`, `semantic_cache`, `scene_cache`, `agent_classification_cache` (all `BoundedCache` from utils).
- **Dependencies:** `openai`, `cekg_pipeline.schemas` (ExtractionError, CEKEvent), `cekg_pipeline.utils` (BoundedCache, _hash_for_cache), `cekg_pipeline.config` (CACHE_MAX_SIZE).

---

### 3.6 `cekg_pipeline/utils.py`

- **Responsibility:** Shared utilities and DAG validation.
- **Key:**
  - `BoundedCache` — async LRU cache used by llm_service.
  - `_make_id(prefix)` — unique ID with prefix.
  - `_hash_for_cache(text, model)` — cache key.
  - `_escape_cypher_string(s)`, `_truncate_safe(text, max_length)` — safe Cypher/export strings.
  - `_normalize_weights(factors)` — normalize why-factor weights.
  - `DAGValidator` — `add_events(events)`, `add_edge(cause_id, effect_id)` (returns False if cycle or invalid), `validate_dag()`, `get_stats()`.
- **Dependencies:** `cekg_pipeline.schemas` (CEKEvent for type hints).

---

### 3.7 `cekg_pipeline/ontology_loader.py`

- **Responsibility:** Load and validate event types, relation types, agent types, place/time types from a JSON schema file (or defaults).
- **Key:** `OntologyManager(schema_path)`: loads from `EventTypeDictionary` or `event_types`, `RelationTypeDictionary` or `RelationTypeDictionary_Truby`/`_McKee`, `AgentTypeDictionary`, `PlaceTypeDictionary`, `TimeTypeDictionary`. Methods: `get_event_type_names()`, `get_relation_type_names(theory)`, `get_agent_type_names(theory)`, `validate_event_type`, `validate_relation_type`, `validate_agent_type`, `get_relation_directionality`. Singleton: `get_ontology_manager(schema_path)`.
- **Dependencies:** `json`, `os`, `re`, `typing`, `dataclasses`.

---

### 3.8 `cekg_pipeline/coreference_resolver.py`

- **Responsibility:** Resolve pronouns and nicknames to canonical character names; filter invalid names (pronouns, generic descriptors).
- **Key:** `CoreferenceResolver`: `is_valid_character_name(name)`, `normalize_character_name(name)`, `register_character(canonical_name, aliases)`, `resolve(mention, context)` → canonical or None, `batch_resolve(mentions)`, `learn_from_cooccurrence(event_actors)`. Singleton: `get_resolver()`.
- **Used by:** pipeline in `_parse_event_json_data()` for actors and patients before creating EventProducesEntity and entity IDs.
- **Dependencies:** `re`, `typing`, `collections.defaultdict`.

---

### 3.9 `cekg_pipeline/graph_builder.py`

- **Responsibility:** Context propagation and entity–event link creation; no LLM calls.
- **Key functions:**
  - `_generate_entity_id(name, type_prefix, event_id, graph_model)` — star vs chain ID (canonical vs per-event).
  - `propagate_context_attributes(events)` — propagate location and time along sequence (in-place).
  - `propagate_context(events, event_produces, entity_occurrences, graph_model)` — propagate actors/whyfactors; returns `(newly_produced_list, updated_entity_occurrences)`.
  - `create_entity_to_event_links(entity_occurrences, event_produces, graph_model)` — returns `List[EntityPointsToEvent]` (star: entity → own event; chain: entity → next event).
- **Dependencies:** `cekg_pipeline.schemas`, `cekg_pipeline.utils` (_make_id).

---

### 3.10 `cekg_pipeline/optimized_linking.py`

- **Responsibility:** Smart candidate pair generation for causal linking (reduce O(N²) to a capped set of high-value pairs).
- **Key:** `IntelligentCausalLinker(use_embeddings)`: `get_candidate_pairs(events, entity_occurrences, max_pairs)` — merges entity-guided, temporal-window, chapter-transition, semantic-similarity (optional), and narrative-peak strategies; returns list of `(cause_id, effect_id)`. Used as **fallback** when dynamic context is disabled or unavailable.
- **Standalone async:** `intelligent_long_range_linking(...)` — gets candidates, runs bulk assessment via provided `assess_pairs_bulk_func`, returns `(causal_links, count)`.
- **Dependencies:** `cekg_pipeline.schemas` (CausalLink), optional `sentence_transformers`, `numpy`.

---

### 3.10b `cekg_pipeline/dynamic_context.py`

- **Responsibility:** Dynamic context windows for candidate pair generation. Uses the **local thematic engine** (embeddings) as a calculation machine to find long-shot event pairs (e.g. events 99 and 3023) with high thematic similarity, instead of fixed first/last-N windows.
- **Key functions:**
  - `get_long_shot_pairs_double_sliding(events, event_map, similarity_threshold=0.95, ...)` — two sliding windows moving in opposite directions (left forward, right backward); keeps only pairs with cosine similarity ≥ threshold to minimize API calls.
  - `get_local_and_scene_pairs(events, scenes=None, local_window=5)` — always includes chronologically adjacent pairs, pairs within `local_window` distance, and (when `scenes` is provided) all pairs within the same scene (`included_event_ids`).
  - `get_dynamic_context_candidate_pairs(events, entity_occurrences, scenes=None, thematic_threshold=0.95, max_pairs=50000, ...)` — main entry: merges long-shot (double sliding + 0.95 thematic), local/scene, and optional entity-guided pairs; returns list of `(cause_id, effect_id)` for the remote LLM to label.
- **CLI:** `--thematic-threshold` (default 0.95), `--no-dynamic-context` to disable and use legacy `optimized_linking` only.
- **Dependencies:** optional `sentence_transformers`, `numpy`.

---

### 3.11 `cekg_pipeline/integrated_semantic.py`

- **Responsibility:** Combined causal + semantic assessment in one LLM call; local embedding-based semantic links; merge with LLM semantic links.
- **Key functions:**
  - `assess_pairs_integrated(pairs_batch, model, client, causal_relations, llm_call_function)` — returns `(causal_results_per_pair, semantic_results_per_pair)`.
  - `process_pairs_with_semantic_linking(pairs_with_text, model, client, relation_ontology, theory_name, dag_validator, ontology_validator, llm_call_function, max_concurrent_calls, bulk_size)` — batches pairs, calls `assess_pairs_integrated`, builds `CausalLink` and `SemanticLink` lists; returns `(all_causal, all_semantic)`.
  - `detect_semantic_links_locally(events, window, similarity_threshold)` — uses `EMBEDDING_MODEL` (SentenceTransformer) to add thematic_similarity links.
  - `create_hybrid_semantic_links(events, llm_semantic_links)` — merges local + LLM semantic links.
- **Dependencies:** `cekg_pipeline.schemas` (CausalLink, SemanticLink), `cekg_pipeline.utils` (_make_id, _hash_for_cache), `cekg_pipeline.llm_service` (assessment_cache), optional `sentence_transformers`, `torch`.

---

### 3.12 `cekg_pipeline/pipeline.py`

- **Responsibility:** Orchestrates the full pipeline: checkpointing, text load/split, event extraction (chunked, with retries), parsing with coreference, context propagation, optional agent classification and scene grouping, causal + semantic linking, export (JSON, CSV, Cypher).
- **Key class:** `CEKGPreprocessor(openai_model, schema_path, checkpoint_dir, enable_checkpoints)`. Key method: `run_async(text_path, out_json, out_cypher, out_csv_dir, max_chapters, graph_model, enable_mixed_theory, enable_agent_classification, enable_scene_grouping, enable_confidence_calibration, enable_semantic_linking, max_concurrent_calls, max_long_range_pairs, chunk_size, resume_from_checkpoint)` — runs the six stages (with optional resume), then exports and returns stats + in-memory results.
- **Serialization helpers:** `_serialize_events`, `_deserialize_events`, `_serialize_links`, `_deserialize_event_produces`, `_deserialize_entity_points_to`, `_deserialize_causal_links`, `_deserialize_semantic_links`, `_deserialize_scenes` (for checkpoints).
- **Other:** `_parse_event_json_data()` (uses coreference_resolver, builds CEKEvent and EventProducesEntity); `_process_chunk_with_retry()`; `_process_chapter_chunked()`; `_calculate_calibrated_confidence()`; `_infer_theory_from_event_type()`; `_classify_agent_types()`; `_generate_scenes_optimized()`; `_integrated_causal_and_semantic_linking()`.
- **Dependencies:** `cekg_pipeline` (config, schemas, utils, text_processor, llm_service, graph_builder, graph_mapper, exporters), `ontology_loader`, `optimized_linking`, `integrated_semantic`, `coreference_resolver`, `checkpoint_manager`; optional `sentence_transformers`.

---

### 3.13 `cekg_pipeline/graph_mapper.py`

- **Responsibility:** Map pipeline domain objects to generic nodes and relationships for export.
- **Key:** `map_to_generic_graph(events, event_produces, entity_points_to, causal_links, graph_model, semantic_links, scenes, agent_classifications)` → `(List[GenericNode], List[GenericRelationship])`. Handles star vs chain (canonical entity IDs vs event-scoped); adds Event, Agent/WhyFactor, Scene nodes; INCLUDES, HAS_PARTICIPANT, HAS_MOTIVATION, ACTS_IN/AFFECTED_IN/MOTIVATES, PRODUCES (chain), FOLLOWS, causal relation types, semantic relation types.
- **Dependencies:** `cekg_pipeline.schemas`, `cekg_pipeline.utils` (_escape_cypher_string, _truncate_safe).

---

### 3.14 `cekg_pipeline/exporters.py`

- **Responsibility:** Write JSON-LD, CSV set for Neo4j, and Neo4j Cypher script.
- **Key functions:**
  - `build_jsonld(events, event_produces, entity_points_to, causal_links)` — returns a dict with `@graph` (events, EventProducesEntity, EntityPointsToEvent, CausalEdge).
  - `export_json(path, data)` — write JSON file.
  - `export_csv(out_dir, events, event_produces, entity_points_to, causal_links, semantic_links, scenes, graph_model)` — writes events, agents, places, whyfactors, produces_*, acts_in, affected_in, motivates, hosts, follows, causes, scenes, scene_includes_event, semantic_links CSVs; returns dict of written paths. Uses pandas if available.
  - `export_neo4j_cypher(path, nodes, relationships, batch_size)` — writes MERGE-based Cypher (nodes by label, then relationships) to `path` (suffix normalized to `.txt`); includes constraint/index creation.
- **Helpers:** `_needs_backtick_escaping`, `_escape_identifier`, `_escape_cypher_value`, `_format_cypher_properties` for Neo4j-safe output.
- **Dependencies:** `cekg_pipeline.schemas`, `cekg_pipeline.utils` (_truncate_safe); `pandas` optional; `os`, `json`, `csv`.

---

### 3.15 `cekg_pipeline/checkpoint_manager.py`

- **Responsibility:** Save/load pipeline stage data for resume and debugging.
- **Key:** `CheckpointManager(checkpoint_dir, run_id)`: `save_checkpoint(stage, data, description)`, `load_checkpoint(stage)` (returns data dict or None), `has_checkpoint(stage)`, `list_checkpoints()`, `get_last_checkpoint()`, `clear_checkpoint(stage)`, `clear_all()`, `get_progress_summary()`. Data is pickled (with hash); a JSON snapshot is also written for inspection.
- **Dependencies:** `os`, `json`, `pickle`, `hashlib`, `pathlib`, `dataclasses.asdict`.

---

### 3.16 `neo4j_csv/consolidate.py`

- **Responsibility:** **Downstream of the main pipeline.** Reads the CSV files produced by the pipeline (in `neo4j_csv/` or similar), normalizes column names (e.g. :ID, START_ID), and outputs consolidated node/edge files (e.g. for Cosmograph or other tools). It is **not** called by `main.py` or the CEKG pipeline; it is a separate utility that consumes pipeline output. Note: it expects an edge file named `causal_links.csv`; the pipeline writes `causes.csv`. You may need to rename or adjust the script to match pipeline output.
- **Dependencies:** `pandas`, `os`.

---

## 4. How to Modify & Extend

### 4.1 Adding a New Data Source (New Input Format)

- **Where:** Input is currently a single text file path. The flow starts in `main.py` (resolve `input_path`) and then `pipeline.run_async(text_path=...)`.
- **What to change:**
  - **New file format (e.g. PDF, HTML):** Add a loader in a new module or in `text_processor.py` (e.g. `load_text_from_pdf(path)`) that returns a single string. In `main.py` or in the first stage of `pipeline.run_async`, call that loader instead of `text_processor.load_text(text_path)` when the path has a given extension or flag.
  - **Pre-split chapters:** If your source is already chapter-separated (e.g. one file per chapter), you can bypass `split_chapters` by building `chapters: List[tuple[int, str]]` yourself and saving that in a checkpoint or passing it into the pipeline (would require a small refactor so stage 1 can accept either `text_path` or a precomputed `chapters` list).

### 4.2 Changing the Graph Schema (Nodes/Edges)

- **Event types, relation types, agent types, place/time types:** Edit the **schema file** passed via `--schema-path` (e.g. `schema.json`). The pipeline uses `cekg_pipeline/ontology_loader.py`, which reads:
  - **EventTypeDictionary** or **event_types**
  - **RelationTypeDictionary** or **RelationTypeDictionary_Truby** + **RelationTypeDictionary_McKee**
  - **AgentTypeDictionary**
  - **PlaceTypeDictionary**, **TimeTypeDictionary**
  Add or remove entries there; the loader and validators will pick them up on next run.

- **Node/edge structure in the exported graph:**  
  - **Domain model (what is stored in memory):** `cekg_pipeline/schemas.py` — add or extend dataclasses (e.g. new relationship type on EventProducesEntity or a new link type). Then update the code that creates these objects (e.g. `pipeline._parse_event_json_data`, `graph_builder`, `pipeline._integrated_causal_and_semantic_linking`).
  - **Export to Neo4j/CSV:** `cekg_pipeline/graph_mapper.py` (how domain objects become GenericNode/GenericRelationship) and `cekg_pipeline/exporters.py` (how CSV columns and Cypher properties are written). To add a new node label or relationship type, add the corresponding creation in `map_to_generic_graph()` and, if needed, new CSV writers or Cypher patterns in `export_csv()` / `export_neo4j_cypher()`.

### 4.3 Adding a New Pipeline Stage

- Implement the stage (e.g. new function or async method that takes current `all_events`, `all_produces`, etc., and returns updated or new structures).
- In `cekg_pipeline/pipeline.py`, inside `run_async()`:
  - Add a checkpoint key (e.g. `"my_stage"`).
  - After the stage that produces the inputs to your stage, add: if resume and checkpoint exists, load and deserialize; else run your stage, then save checkpoint (using existing serialization helpers if applicable).
- If your stage needs new CLI flags, add them in `main.py` and pass them into `run_async(...)`.

### 4.4 Changing LLM Prompts or Models

- **Prompts:** `cekg_pipeline/llm_service.py` — constants `PROMPT_EVENT_EXTRACTION`, `PROMPT_CAUSAL_BULK`, `PROMPT_AGENT_CLASS`, `PROMPT_SCENE`. Integrated causal+semantic prompt is in `cekg_pipeline/integrated_semantic.py` (`PROMPT_INTEGRATED_ASSESSMENT`).
- **Model:** `main.py` uses `--openai-model` (default from `config.OPENAI_MODEL`). Override via CLI or set `OPENAI_MODEL` in `.env` / `config.py`.

---

## 5. Quick Start

### 5.1 Requirements

- **Python:** 3.x (code uses `asyncio`, `dataclasses`, type hints).
- **Dependencies** (from project root; see `requirements.txt`):
  - `openai`
  - `pandas`
  - `python-dotenv`
  - `sentence-transformers` (optional; used for semantic/linking embeddings and local semantic links; pipeline runs without it with a warning).

Install:

```bash
pip install -r requirements.txt
```

- **Environment:** Set `OPENAI_API_KEY` (required). Optional: `OPENAI_MODEL` (default in code: `gpt-4o-mini`).

```bash
export OPENAI_API_KEY='sk-...'
```

### 5.2 Run the Full Pipeline

From the project root:

```bash
# Full run (all features: scene grouping, agent classification, confidence calibration, semantic linking, mixed theory)
python main.py --input "Great Expectations.txt" --full

# Outputs (default paths):
#   JSON:  ./ge_preprocessed.json
#   Cypher: ./ge_import.txt
#   CSV:   ./neo4j_csv/
```

**Resume from last checkpoint:**

```bash
python main.py --input "Great Expectations.txt" --full --resume
```

**Clear checkpoints and start fresh:**

```bash
python main.py --input "Great Expectations.txt" --full --clear-checkpoints
```

**Fast mode (fewer features, lower cost):**

```bash
python main.py --input "Great Expectations.txt" --fast
```

**Custom paths and limits:**

```bash
python main.py --input "novels/MyNovel.txt" \
  --out-json ./out/ge_preprocessed.json \
  --out-cypher ./out/ge_import.txt \
  --out-csv ./out/neo4j_csv \
  --max-chapters 5 \
  --max-pairs 3000 \
  --chunk-size 3000 \
  --graph-model star \
  --full
```

**Use a custom ontology schema:**

```bash
python main.py --input "Great Expectations.txt" --full --schema-path schema.json
```

**List checkpoint status (no run):**

```bash
python main.py --input "Great Expectations.txt" --list-checkpoints
```

### 5.3 Optional: Post-Process CSVs

The script `neo4j_csv/consolidate.py` expects to be run in a directory that already contains the pipeline CSV output (e.g. `events.csv`, `agents.csv`, `causes.csv`, etc.). Run it from that directory (e.g. `neo4j_csv/`) to produce consolidated node/edge files; it is independent of `main.py`.

---

*End of Developer Manual. All descriptions are derived from the current codebase.*
