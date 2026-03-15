[cite_start]You are an expert Python developer tasked with implementing the "CEKG Thematic Layer v2" for the Narrative Causal Graph pipeline[cite: 276]. [cite_start]The objective is to annotate events with participation in structural themes based on their local causal context, rather than interpreting broad literary meaning[cite: 278, 285].

Please implement this layer strictly adhering to the following specifications:

### 1. Schema Modifications (`cekg_pipeline/schemas.py`)
[cite_start]Update the existing dataclasses to support the new layer[cite: 462, 465]:
* [cite_start]**`CEKEvent`**: Add the following fields[cite: 463]:
    * [cite_start]`theme_annotations: Dict[str, Any] = field(default_factory=dict)` [cite: 464]
    * [cite_start]`scene_id: Optional[str] = None` [cite: 464]
* [cite_start]**`CausalLink`**: Add the following field[cite: 466]:
    * [cite_start]`edge_supertype: Optional[str] = None` [cite: 467]

### 2. New Module: Theme Annotation (`cekg_pipeline/theme_annotation.py`)
Create a new file to handle the thematic logic. [cite_start]It must include the following constants[cite: 484, 485]:

```python
THEME_SET = [
    [cite_start]"POWER", # authority, command, hierarchy [cite: 515, 516]
    [cite_start]"WEALTH", # transfer or control of material resources [cite: 517]
    [cite_start]"KINSHIP", # family or household relations [cite: 518, 519]
    [cite_start]"JUSTICE", # rule violation, accusation, punishment [cite: 520]
    [cite_start]"KNOWLEDGE" # revelation or concealment of information [cite: 520]
[cite_start]] # (Ensure previous themes like CLASS_MOBILITY or MONEY_PROPERTY are replaced [cite: 287, 288, 289])

ROLE_SET = [
    [cite_start]"initiating", # starts a theme chain [cite: 371, 379]
    [cite_start]"enabling", # provides resources or conditions [cite: 372, 381]
    [cite_start]"constraining", # imposes limits or pressure [cite: 373, 382]
    [cite_start]"mediating", # transmits influence or information [cite: 374, 384]
    [cite_start]"escalating", # increases stakes [cite: 375, 386]
    [cite_start]"resolving", # stabilizes or closes the chain [cite: 376, 388]
    [cite_start]"revealing" # discloses hidden causes or identities [cite: 377, 389]
]

INVOLVEMENT_SET = [
    [cite_start]"direct", # explicitly instantiates the theme mechanism [cite: 361, 362]
    [cite_start]"indirect", # supports or mediates a theme-related causal chain [cite: 363, 364]
    [cite_start]"latent", # contains signals that may become theme-relevant later [cite: 365, 366]
    [cite_start]"none" # no evidence of theme involvement [cite: 367, 368]
]

```

#### Functions to Implement in this module:

1. 
**`assign_edge_supertypes(causal_links)`**: Map fine-grained causal relations to `edge_supertype` to simplify querying. Use mappings like: `CAUSES` -> `CAUSAL_PRODUCTION`, `ENABLES` -> `CAUSAL_PRODUCTION`, `PREVENTS` -> `CAUSAL_CONSTRAINT`, `REVEALS` -> `REVELATION_EPISTEMIC`, `INFORMS` -> `MEDIATION_TRANSFER`.


2. 
**`attach_scene_ids_to_events(events, scenes)`**: Map the `scene.id` to the corresponding `CEKEvent` objects.


3. **`build_local_causal_context(event, causal_links, all_events)`**: Generate a context payload for the LLM. It MUST include: `event_id`, `event_text` (from raw_description), `actors`, `patients`, `chapter`, `scene_id`, up to 2 immediate `causes`, and up to 2 immediate `effects`. Theme participation must be evaluated relative to these neighboring causal events.


4. 
**`apply_theme_bridge_rule(events, causal_links)`**: A deterministic post-processing step to preserve micro-events. If an event E has `theme.involvement == none` , but an adjacent cause or effect event has `theme.involvement == direct` , upgrade event E's annotation for that theme to `involvement = indirect` and `role = mediating`.



### 3. LLM Service Updates (`cekg_pipeline/llm_service.py`)

Add a dedicated cache for theme calls and implement an async function (`annotate_single_event_theme`) utilizing the following prompt template:

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

```

Note: Make sure the output structure for each theme contains keys for `involvement`, `role`, `evidence`, `signals`, and `confidence`. Remember the constraints: evidence must be <=2 sentences , signals must be short event cues (not interpretation) , and role must be null if involvement is none.

### 4. Pipeline Integration (`cekg_pipeline/pipeline.py`)

Insert the new "Theme Annotation" stage (stage 6) exactly after "Scene Grouping" (stage 5) and before "Export" (stage 7). Ensure the main `annotate_event_themes` orchestrator calls the functions in this order: attach scene ids, assign edge supertypes, build causal context, annotate via LLM, and apply the theme-bridge rule. Use `asyncio.gather` to perform these LLM calls efficiently.

### 5. Exporters Updates (`cekg_pipeline/exporters.py`)

Ensure both structures can be queried independently. To do this, modify the CSV export functions:

* 
**`events.csv`**: Must include `scene_id` and `theme_annotations` (serialize the dictionary to a JSON string).


* 
**`causes.csv` (causal_links)**: Must include the new `edge_supertype` field.


* Update `build_jsonld` similarly to pass these properties through.
