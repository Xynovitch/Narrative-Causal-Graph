You are an expert Python developer tasked with implementing the "Thematic Layer v2" for the CEKG (Causal Event Knowledge Graph) pipeline. I need you to create a new module and update several existing files to annotate events with structural literary themes based on their local causal context.

### Architectural Overview & Rules
We are moving away from broad literary interpretation to identifying an event's causal participation in theme-related chains. 
* **New V2 Theme Set:** `POWER`, `WEALTH`, `KINSHIP`, `JUSTICE`, `KNOWLEDGE`. (Do NOT use any older themes like CLASS_MOBILITY).
* **Roles:** `initiating`, `enabling`, `constraining`, `mediating`, `escalating`, `resolving`, `revealing`.
* **Involvement Levels:** `direct`, `indirect`, `latent`, `none`.
* **Theme-Bridge Rule:** A deterministic post-processing step to preserve micro-events. If an event has `involvement="none"` but an adjacent cause or effect has `involvement="direct"`, upgrade the current event to `involvement="indirect"` and `role="mediating"`.

### Step-by-Step Implementation Guide

#### 1. Update `cekg_pipeline/schemas.py`
* Modify `CEKEvent` to add two new fields:
  - `theme_annotations: Dict[str, Any] = field(default_factory=dict)`
  - `scene_id: Optional[str] = None`
* Modify `CausalLink` to add one new field:
  - `edge_supertype: Optional[str] = None`

#### 2. Update `cekg_pipeline/llm_service.py`
* Create a dedicated cache near the top: `theme_annotation_cache = BoundedCache(max_size=5000)`.
* Add the following exact LLM prompt template:
  ```python
  PROMPT_THEME_ANNOTATION = """You are annotating participation of narrative events in structural theme chains.
  IMPORTANT:
  You are NOT interpreting literary meaning.
  You are identifying whether the event participates in a theme-related causal mechanism.
  Themes:
  POWER authority, command, hierarchy
  WEALTH transfer or control of material resources
  KINSHIP family or household relations
  JUSTICE rule violation, accusation, punishment
  KNOWLEDGE revelation or concealment of information

  Use only the event and its nearby causal context.
  Return JSON only.
  Required structure:
  {{
    "event_id": "{event_id}",
    "theme_annotations": {{
      "POWER": {{...}},
      "WEALTH": {{...}},
      "KINSHIP": {{...}},
      "JUSTICE": {{...}},
      "KNOWLEDGE": {{...}}
    }}
  }}
  Rules:
  Use involvement = direct | indirect | latent | none
  If involvement none, role must be null
  Evidence must be <=2 short sentences
  Signals must be event cues, not interpretation
  Use local causal context when deciding indirect roles"""

* Implement an async function `annotate_single_event_theme(event_context_json, model, client)` that formats this prompt and uses the existing `_async_llm_json_call` (make sure to use the `theme_annotation_cache`).

#### 3. Create `cekg_pipeline/theme_annotation.py`

Create this new file to handle the logic. It must include:

* The Constants: `THEME_SET`, `ROLE_SET`, and `INVOLVEMENT_SET` matching the V2 specs.
* A `FINE_TO_SUPERTYPE` dictionary mapping fine-grained relations to broader categories (e.g., `"CAUSES": "CAUSAL_PRODUCTION"`, `"ENABLES": "CAUSAL_PRODUCTION"`, `"PREVENTS": "CAUSAL_CONSTRAINT"`, `"REVEALS": "REVELATION_EPISTEMIC"`, `"INFORMS": "MEDIATION_TRANSFER"`). Feel free to infer mappings for other relations based on these examples.
* `assign_edge_supertypes(causal_links)`: Loops through links and assigns the `edge_supertype` based on the dictionary.
* `attach_scene_ids_to_events(events, scenes)`: Maps `scene.id` to the `scene_id` property of each `CEKEvent` included in that scene.
* `build_local_causal_context(event, causal_links, all_events)`: Creates a dict for the LLM containing the event text, actors, patients, why_factors, scene_id, chapter, and up to 2 immediate causes and 2 immediate effects (resolved from the causal links).
* `apply_theme_bridge_rule(events, causal_links)`: Implements the deterministic bridge rule mentioned in the overview.
* `annotate_event_themes(events, causal_links, scenes, model, client)`: The main entry point that runs the above steps in order, using `asyncio.gather` to batch the LLM calls to `annotate_single_event_theme` for efficiency.

#### 4. Update `cekg_pipeline/pipeline.py`

* In `run_async`, insert the new "Theme Annotation" stage right after Stage 5 (Scene Grouping).
* Call the new `annotate_event_themes` pipeline function.
* Update the `checkpoint_manager` block to save and load this new stage (e.g., `checkpoint_mgr.has_checkpoint("theme_annotation")`).

#### 5. Update `cekg_pipeline/exporters.py`

* In `export_csv`:
* Add `scene_id` and `theme_annotations` (serialized as a JSON string using `json.dumps`) to the `events_rows` dictionaries.
* Add `edge_supertype` to the `causes_rows` dictionaries.


* In `build_jsonld`: Include these new properties in the respective Event and CausalEdge outputs.
* Ensure `export_neo4j_cypher` is unaffected or natively picks up the dynamic properties correctly via the generic graph mapper.

Review the existing code style, ensure all imports are properly configured, and implement these changes carefully without removing existing graph functionality.

```