# Causal Event Knowledge Graph (CEKG)

A Python pipeline that extracts a structured causal knowledge graph from narrative text (novels). Given a `.txt` file, it produces a directed graph of events, characters, and causal/thematic relationships — grounded in McKee and Truby narrative theory — exportable to Neo4j, CSV, and JSON-LD.

---

## What It Produces

Every node and edge answers a specific narrative question:

| Element | Type | Question Answered |
|---------|------|-------------------|
| `Event` | Node | What happened? |
| `Agent` | Node | Who acted or was affected? |
| `WhyFactor` | Node | What motivated it? |
| `Place` | Node | Where did it happen? |
| `Scene` | Node | What scene did it belong to? |
| `FOLLOWS` | Edge | When did this happen relative to that? |
| `CAUSES` | Edge | Did this causally necessitate that? |
| `THEMATIC` | Edge | What structural theme connects these two events? |
| `ACTS_IN` | Edge | Which agent acted in this event? |
| `AFFECTED_IN` | Edge | Which agent was affected? |
| `MOTIVATES` | Edge | What motivated this event? |
| `HOSTS` | Edge | Where did this event take place? |

### Graph Snapshot — *Great Expectations* (Dickens)

| Metric | Value |
|--------|-------|
| Events extracted | ~4,700 |
| Characters identified | ~100 |
| Causal links (McKee + Truby) | ~3,200 |
| Thematic links | derived from theme annotations |
| Scenes | ~300 |
| DAG valid | Yes |
| Full-mode cost | ~$1.35–$5.00 |

---

## Pipeline Architecture

```
Raw .txt
  └─ [Stage 1] split_chapters()
  └─ [Stage 2] LLM event extraction  (per chapter, chunked)
  └─ [Stage 3] Context propagation   (location/time/actors carry forward)
  └─ [Stage 4] Agent classification  (47 narrative role types from schema.json)
  └─ [Stage 5] Scene grouping        (cluster events by spatial/temporal/thematic coherence)
  └─ [Stage 6] Causal linking        (candidate pair generation → LLM causal assessment)
  └─ [Stage 7] Theme annotation      (LLM tags each event with POWER/WEALTH/KINSHIP/JUSTICE/KNOWLEDGE)
                + Theme-Bridge Rule  (propagate involvement to causal neighbours)
                + build_thematic_links()  (THEMATIC edges from co-participation)
  └─ Export: JSON-LD, Neo4j Cypher, CSV
```

### Candidate Pair Generation

Stage 6 does not evaluate all O(N²) event pairs. Instead it uses two strategies:

1. **Dynamic context windows** (default): combines a sliding window within chapters with long-range pairs discovered via cosine similarity of event embeddings (`thematic_threshold=0.95`). Caps at `--max-pairs`.
2. **Fallback (`IntelligentCausalLinker`)**: entity co-occurrence + temporal proximity scoring.

### Thematic Links

THEMATIC edges are built deterministically from LLM theme annotations — no additional API calls. Rules:
- Both events must have involvement of `"direct"` or `"indirect"` for the same theme.
- At least one must be `"direct"` (prevents indirect↔indirect noise).
- Confidence = min(source confidence, target confidence) ≥ 0.5.

This replaces the previous `SemanticLink` system (cosine-similarity edges), which produced excessive low-quality edges and had no grounding in the narrative ontology.

---

## Setup

```bash
pip install -r requirements.txt
# openai, pandas, python-dotenv, sentence-transformers
```

Create `.env` in the project root:

```
OPENAI_API_KEY="sk-proj-..."
OPENAI_MODEL="gpt-4o-mini"
```

Place your `schema.json` in the project root — it will be auto-detected. The schema defines 47 agent types, causal relation types for both McKee and Truby theories, event type ontology, and place/time type vocabularies.

---

## Usage

```bash
# Full mode — all features, 25,000 candidate pairs (~$3–5/novel)
python main.py --input "Great Expectations.txt" --full

# Fast mode — no scene/agent classification, 3,000 pairs (~$1–2/novel)
python main.py --input "Great Expectations.txt" --fast

# Resume an interrupted run
python main.py --input "Great Expectations.txt" --full --resume

# List checkpoint status for a run
python main.py --input "Great Expectations.txt" --list-checkpoints

# Clear checkpoints and start fresh
python main.py --input "Great Expectations.txt" --full --clear-checkpoints

# Custom flags
python main.py --input novel.txt \
  --max-pairs 10000 \
  --thematic-threshold 0.90 \
  --max-concurrent-calls 15 \
  --chunk-size 3000 \
  --openai-model gpt-4o
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--full` | — | All features, 25,000 candidate pairs |
| `--fast` | — | Minimal features, 3,000 pairs |
| `--max-pairs` | 5,000 | Causal candidate pair cap |
| `--thematic-threshold` | 0.80 | Cosine similarity threshold for long-range pair discovery |
| `--max-concurrent-calls` | 10 | Parallel LLM calls |
| `--chunk-size` | 3,000 | Characters per extraction chunk |
| `--schema-path` | auto | Path to `schema.json` ontology |
| `--resume` | false | Resume from last checkpoint |
| `--no-dynamic-context` | false | Use legacy pair generation instead |
| `--disable-mixed-theory` | false | McKee-only (no Truby) |

---

## Output Files

| File | Contents |
|------|----------|
| `ge_preprocessed.json` | Full JSON-LD graph |
| `ge_import.cypher` | Neo4j Cypher import script |
| `neo4j_csv/events.csv` | Event nodes with theme annotations |
| `neo4j_csv/causes.csv` | Causal edges with relation type, mechanism, confidence |
| `neo4j_csv/thematic_links.csv` | THEMATIC edges with theme, involvement, roles |
| `neo4j_csv/follows.csv` | Chronological ordering edges |
| `neo4j_csv/agents.csv` | Agent nodes |
| `neo4j_csv/scenes.csv` | Scene nodes |
| `neo4j_csv/acts_in.csv` | Agent → Event participation |
| `neo4j_csv/affected_in.csv` | Patient → Event involvement |
| `neo4j_csv/motivates.csv` | WhyFactor → Event motivation |

---

## Cost Estimates

Using `gpt-4o-mini` (default):

| Mode | Novel Size | Approximate Cost |
|------|------------|-----------------|
| Fast | 400 pages | ~$1.00–$1.50 |
| Full | 400 pages | ~$3.00–$5.00 |
| Full | 800 pages | ~$5.00–$8.00 |

The dominant cost driver is event extraction (chapter-level, not paragraph-level — ~93% cheaper than naive approaches). Causal linking adds ~$0.10 per 5,000 pairs. Theme annotation adds ~$0.50–$2.00 depending on novel length.

---

## Key Files

| File | Role |
|------|------|
| `main.py` | CLI entry point |
| `cekg_pipeline/pipeline.py` | Main orchestrator (`CEKGPreprocessor.run_async`) |
| `cekg_pipeline/schemas.py` | All dataclasses (`CEKEvent`, `CausalLink`, `ThematicLink`, `Scene`, …) |
| `cekg_pipeline/llm_service.py` | OpenAI API calls with response caching |
| `cekg_pipeline/integrated_semantic.py` | Causal pair LLM assessment |
| `cekg_pipeline/dynamic_context.py` | Embedding-based long-range pair discovery |
| `cekg_pipeline/theme_annotation.py` | LLM theme annotation + `build_thematic_links()` |
| `cekg_pipeline/optimized_linking.py` | `IntelligentCausalLinker` fallback pair generation |
| `cekg_pipeline/graph_builder.py` | Context propagation and entity–event linking |
| `cekg_pipeline/exporters.py` | JSON-LD, CSV, and Cypher export |
| `cekg_pipeline/graph_mapper.py` | Maps pipeline data to generic graph nodes/edges |
| `cekg_pipeline/ontology_loader.py` | Loads `schema.json` |
| `cekg_pipeline/checkpoint_manager.py` | Pickle-based checkpoint save/resume |
| `cekg_pipeline/config.py` | `.env` loader, batch sizes, cache config |
| `cekg_pipeline/utils.py` | `DAGValidator`, `BoundedCache`, ID generation |
| `schema.json` | Full ontology: 47 agent types, causal relation types, event types |

---

## Checkpoints

Each pipeline run saves checkpoints under `./checkpoints/<run_id>/` as pickle files after stages: `text_split`, `extraction`, `context_propagation`, `agent_classification`, `scenes`, `linking`, `theme_annotation`.

Use `--resume` to continue an interrupted run without re-running completed stages or spending API credits again.

---

## Ontology (`schema.json`)

The schema defines the full narrative vocabulary:

- **47 agent types** — e.g. `PROTAGONIST_HERO`, `MORAL_ANTAGONIST`, `REVELATION_GIVER`, `GHOST_BEARER`, `TURNING_POINT_DRIVER`
- **McKee causal relations** — e.g. `DIRECT_CAUSE`, `ENABLES`, `PREVENTS`, `MORAL_CHALLENGE`, `ESCALATES`, `RESOLVES`
- **Truby causal relations** — e.g. `TRIGGERS`, `CONCEALS`, `REDEEMS`, `DELEGATES`
- **5 structural themes** — `POWER`, `WEALTH`, `KINSHIP`, `JUSTICE`, `KNOWLEDGE`
- **Involvement levels** — `direct`, `indirect`, `latent`, `none`
- **Narrative roles** — `initiating`, `enabling`, `constraining`, `mediating`, `escalating`, `resolving`, `revealing`

Place `schema.json` in the project root and it will be auto-detected on every run.

---

## Changelog

### Recent Changes

**SemanticLink → ThematicLink (architecture refactor)**
- Removed `SemanticLink` edge type entirely. It was generated by cosine similarity of event embeddings and produced large numbers of low-quality edges (82,680 at threshold 0.55) with no grounding in narrative structure.
- Added `ThematicLink` edge type. THEMATIC edges connect events that co-participate in the same structural theme (POWER/WEALTH/KINSHIP/JUSTICE/KNOWLEDGE) with at least one event having `"direct"` involvement. Derived deterministically from LLM theme annotations — zero additional API cost.
- Removed `detect_semantic_links_locally`, `create_hybrid_semantic_links`, and the dual causal+semantic LLM prompt. The causal assessment prompt is now causal-only, which is simpler and more focused.
- Output: `thematic_links.csv` (was `semantic_links.csv`) with columns `theme`, `source_involvement`, `target_involvement`, `source_role`, `target_role`, `confidence`.

**`--full` mode max-pairs fix**
- `--full` now defaults to 25,000 candidate pairs (was incorrectly using the 5,000 default). Override with `--max-pairs`.

**Agent classification fix**
- `schema.json` is now auto-detected in the project root when `--schema-path` is not specified. This enables all 47 agent types from the schema rather than the 5 built-in defaults.

**Removed `--semantic-threshold` and `--enable/disable-semantic-linking` flags**
- These flags no longer exist. Thematic linking is always on and costs nothing.

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
