# CEKG — Response to 0326 Feedback

**Date:** 2026-04-28
**Source feedback:** `Causal Graph(0326) Feedback 3350f3154e82801c9afed0c75d5eaa55.md`
**Author:** Pipeline maintenance pass

---

## 1. Feedback Categorization

The reviewer's notes were grouped into four buckets. Each row below maps a
specific complaint in the feedback file to the change(s) that address it.

| # | Bucket | Concrete complaint (paraphrased) | Fix bucket |
|---|---|---|---|
| **A1** | Theme annotation | `confidence` values were null/empty, so theme annotations couldn't be filtered or compared by confidence. | Prompt + parser |
| **A2** | Theme annotation | Theme-Bridge Rule upgraded events to `indirect / mediating` with **no evidence**, so the upgrade couldn't be explained. | Bridge rule |
| **A3** | Theme annotation | Surface verbs ("replies", "asks", "swear", "sternly asks") were enough to activate POWER and KNOWLEDGE on events whose **core mechanism** wasn't thematic. | Prompt rewrite |
| **A4** | Theme annotation | The seven-role enum (initiating, enabling, …) was never properly used — most events were `null` or `mediating`. | Prompt rewrite |
| **B1** | Causal relations | Despite a large relation ontology, only ~6 types dominated the output (`EVENT_ENABLES_NEXT`, `EVENT_REINFORCEMENT`, `COMPASSION_TRIGGER`, `ALLY_DEPENDENCE`, `EMOTIONAL_MANIPULATION`, `CAUSES_REVERSAL`). | Prompt rewrite |
| **C1** | CSV / Neo4j | `theme_annotations` was exported as a JSON-text blob, forcing brittle `CONTAINS` queries. | Export schema |
| **C2** | CSV / Neo4j | Pip / Magwitch / Estella were scattered across many surface aliases ("Pip"/"Philip Pirrip", "the convict"/"Magwitch"/"the unknown man"), so character-centric queries lost coverage. | Coreference seed |
| **C3** | CSV / Neo4j | Event extraction is too granular — a single plot-unit ("Pip helps Magwitch") is split across many micro-events and useful filtering is impossible at event-level only. | Scene rollups |
| **D1** | New feature | "Themes themselves should have edges based on the 5 themes and their initiating/other attributes" — i.e. themes as edge attributes that *build subplots*. | Thematic edges |

---

## 2. Theme Annotation Quality (A1–A4)

### A1 — Confidence is now a real number

`cekg_pipeline/llm_service.py` — `PROMPT_THEME_ANNOTATION` rewritten:

- `confidence` is **required** as a float in `[0.0, 1.0]`, with explicit
  banding guidance:
  - `0.85+` only when the event text *itself* names the structural mechanism.
  - `0.5–0.85` when local causal context supports it but the event text alone
    is weaker.
  - `<0.5` discouraged → prefer `latent` or `none`.
- Special case: when `involvement="none"`, `confidence=0.0`, `role=null`,
  `evidence=""`, `signals=[]`.

`cekg_pipeline/theme_annotation.py` — parser rewritten:

- `None` confidence is coerced to `0.0` (not left as `None`).
- Out-of-range floats are clamped.
- `signals` is always a list (not the previous `""` default).
- Direct/indirect tags with `confidence < 0.4` are auto-demoted to `latent`,
  silencing the over-tagging that the feedback flagged on weak signals.

### A2 — Bridge rule now records evidence

`cekg_pipeline/theme_annotation.py` — `apply_theme_bridge_rule` rewritten:

When an event is upgraded `none → indirect/mediating` because a neighbour
has direct involvement, the rule now stores:

```json
{
  "involvement": "indirect",
  "role": "mediating",
  "evidence": "Bridge: linked to event/<id> via <RELATION_TYPE> which has direct WEALTH involvement.",
  "signals": ["bridge_from:event/<id>", "via:<RELATION_TYPE>"],
  "confidence": <propagated, ~0.6 × neighbour_confidence, capped 0.6>,
  "bridge_source": "event/<id>",
  "bridge_relation": "<RELATION_TYPE>"
}
```

The `event/597439eb` example in the feedback (KINSHIP indirect with empty
evidence) would now contain a populated `evidence` string, the source event
ID, and the relation that justified the upgrade.

### A3 — Stop activating themes on weak verbs

The new `PROMPT_THEME_ANNOTATION` includes per-theme **NOT** clauses, e.g.:

> - **POWER**: explicit authority/command/coercion that constrains another
>   agent's choices. **NOT** POWER: a tone of voice, a polite request,
>   narration of social rank.
> - **KNOWLEDGE**: information is revealed, concealed, learned, or
>   transmitted that *changes what someone can do*. **NOT** KNOWLEDGE:
>   ordinary speech ("replies", "asks") without epistemic shift.

Combined with the auto-demotion of low-confidence direct/indirect tags to
`latent`, this addresses the reviewer's "POWER/KNOWLEDGE on every dialogue
line" complaint.

### A4 — Role distribution

The role enum is now spelled out in the prompt with a one-line definition for
each role and an explicit instruction *not to default to mediating*. The
seven roles (`initiating`, `enabling`, `escalating`, `constraining`,
`mediating`, `revealing`, `resolving`) each get a one-line trigger
description so the model picks the single best fit.

---

## 3. Causal Relation Diversity (B1)

`cekg_pipeline/integrated_semantic.py`:

- `PROMPT_CAUSAL_ASSESSMENT` rewritten with a **selection-guidance block**
  that lists specific psychological / social / moral / epistemic types and
  explicitly instructs the model to *prefer* them over the generic
  `EVENT_ENABLES_NEXT` / `DIRECT_CAUSE` fallbacks.
- The full ontology is now passed to the prompt — previously it was
  truncated to `causal_relations[:15]`, which itself biased the model toward
  the same handful of types.
- Cache key bumped to `causal_v2` / `causal_rag_v2` so old narrow-relation
  cached results don't shadow the new prompt on resume.

---

## 4. Query-ability (C1–C3)

### C1 — Flat per-theme columns

`cekg_pipeline/exporters.py`, `cekg_pipeline/graph_mapper.py`:

The `events.csv` and the `Event` Cypher properties now contain explicit
queryable columns instead of a JSON text blob:

| Column | Type | Example |
|---|---|---|
| `theme_POWER_involvement` | string | `direct` / `indirect` / `latent` / `none` |
| `theme_POWER_role` | string | `initiating` / `enabling` / … / `""` |
| `theme_POWER_confidence` | float | `0.85` |
| `theme_POWER_evidence` | string | short evidence sentence |

(repeated for WEALTH / KINSHIP / JUSTICE / KNOWLEDGE)

The legacy `theme_<T>` short property (= involvement only) is preserved for
backward compatibility, plus a `theme_annotations_raw` JSON column for
debugging. Cypher filters now look like:

```cypher
MATCH (e:Event)
WHERE e.theme_WEALTH_involvement = 'direct'
  AND e.theme_WEALTH_confidence >= 0.6
RETURN e
```

### C2 — Canonical character alias dictionary

`cekg_pipeline/coreference_resolver.py`:

Added a `KNOWN_WORK_ALIASES` dictionary (initially populated for *Great
Expectations*) mapping canonical names to known surface aliases:

```python
"Abel Magwitch": ["Magwitch", "the convict", "convict",
                  "the man", "Provis", "Mr. Provis",
                  "the stranger", "the unknown man",
                  "the runaway"],
"Philip Pirrip": ["Pip", "Pirrip", "young Pip",
                  "the boy", "Mr. Pip", "Handel"],
…
```

A new `seed_from_work(work_title)` method registers every alias before
extraction begins. `pipeline.py` calls it from a new "stage 0" before text
splitting:

```
[resolver] Seeded 40 canonical character aliases from work dictionary.
```

Verified by direct test:

```
'Pip'             -> 'Philip Pirrip'
'the convict'     -> 'Abel Magwitch'
'Magwitch'        -> 'Abel Magwitch'
'the unknown man' -> 'Abel Magwitch'
'Provis'          -> 'Abel Magwitch'
'Joe'             -> 'Joe Gargery'
'Estella'         -> 'Estella Havisham'
'Drummle'         -> 'Bentley Drummle'
```

### C3 — Scene rollups (plot-unit filter)

`cekg_pipeline/graph_mapper.py`, `cekg_pipeline/exporters.py`:

`Scene` nodes / rows now carry per-theme rollups so the reviewer's
"Pip helping Magwitch cluster" can be filtered as a single plot-unit:

| Scene property | Meaning |
|---|---|
| `event_count` | number of events in the scene |
| `participant_count` | number of distinct actor/patient names |
| `theme_<T>_event_count` | events in scene with non-none T involvement |
| `theme_<T>_direct_count` | … with direct involvement |
| `theme_<T>_indirect_count` | … with indirect involvement |
| `theme_<T>_avg_confidence` | mean confidence across participating events |

```cypher
// Find scenes that are dominated by the WEALTH theme
MATCH (s:Scene)
WHERE s.theme_WEALTH_direct_count >= 3
RETURN s
ORDER BY s.theme_WEALTH_avg_confidence DESC
```

---

## 5. New Feature — Thematic Subplot Edges

This is the new feature requested on top of the 0326 review:

> "themes themselves should be given edges based on the 5 themes (WEALTH,
> POWER) etc and their (initiating and other attributes)"

### Design

Themes are **edge properties**, not separate nodes. Two events are connected
by a `THEMATIC_LINK` edge whenever they share an active theme along a
narrative beat. The connected component induced by all edges with the same
`theme` value is the **subplot** for that theme.

```
Event -[:THEMATIC_LINK { theme,
                         source_role, target_role,
                         source_involvement, target_involvement,
                         source_confidence, target_confidence,
                         confidence,            // sqrt(s · t)
                         via,                   // provenance
                         scene_id,
                         sequence_distance }]-> Event
```

### Generation modes

Two complementary modes feed a single deduplicated edge set
(deduplicated on `(source, target, theme)`):

1. **Causal-projected** (`via: "causal:<RELATION_TYPE>"`) — for every
   existing causal link `cause → effect`, if both events have non-none
   involvement of theme T, emit a thematic edge with `theme=T`. The causal
   beat *is* a subplot beat.
2. **Scene-spine** (`via: "scene_spine:<scene_id>"`) — within each scene,
   for each theme T, sort the scene's events by sequence and link
   consecutive theme-active events. Captures subplot continuity that isn't
   marked as causal but still belongs to the theme's spine.

Implementation: `cekg_pipeline/theme_graph.py` (new module).

### Cypher query patterns

```cypher
// Walk the WEALTH subplot in story order
MATCH p = (a:Event)-[r:THEMATIC_LINK*1..50 {theme: 'WEALTH'}]->(b:Event)
RETURN p

// Subplot beats where POWER initiates and the effect escalates
MATCH (a:Event)-[r:THEMATIC_LINK {
   theme: 'POWER',
   source_role: 'initiating',
   target_role: 'escalating'
}]->(b:Event)
RETURN a, r, b

// High-confidence subplot beats backed by the causal graph only
MATCH (a)-[r:THEMATIC_LINK]->(b)
WHERE r.via STARTS WITH 'causal:'
  AND r.confidence >= 0.6
RETURN a, r, b

// All five subplots at once, sliced by theme
MATCH (a:Event)-[r:THEMATIC_LINK]->(b:Event)
RETURN r.theme AS subplot, count(*) AS beats
ORDER BY beats DESC
```

### Smoke-test output

Synthetic case with three events `e1 → e2 → e3` (causal links) and a
`scene_spine` fallback test:

```
4 thematic edges from causal projection:
  e1 -> e2  theme=POWER      initiating/direct -> escalating/direct  conf=0.875  via=causal:EMOTIONAL_TRIGGER
  e1 -> e2  theme=WEALTH     enabling/direct   -> mediating/indirect conf=0.693  via=causal:EMOTIONAL_TRIGGER
  e2 -> e3  theme=WEALTH     mediating/indirect-> resolving/direct   conf=0.693  via=causal:REVEALS
  e2 -> e3  theme=KNOWLEDGE  revealing/direct  -> mediating/direct   conf=0.592  via=causal:REVEALS

scene-spine fallback (e2 has no WEALTH involvement):
  e1 -> e3  theme=WEALTH     initiating -> resolving                 via=scene_spine:s
```

The fallback correctly skips a non-active intermediate event and still
links the two endpoints of the WEALTH subplot.

---

## 6. Files Changed

| File | Status | Purpose |
|---|---|---|
| `cekg_pipeline/llm_service.py` | modified | Strict theme prompt (A1, A3, A4) |
| `cekg_pipeline/theme_annotation.py` | modified | Confidence parser, evidence-recording bridge rule, latent demotion of weak signals (A1, A2, A3) |
| `cekg_pipeline/integrated_semantic.py` | modified | Diversified causal prompt; full ontology surfaced; cache v2 (B1) |
| `cekg_pipeline/coreference_resolver.py` | modified | `KNOWN_WORK_ALIASES` dict + `seed_from_work()` (C2) |
| `cekg_pipeline/exporters.py` | modified | Flat theme columns, thematic-edge CSV, scene rollups, JSON-LD thematic edges, Cypher index (C1, C3, D1) |
| `cekg_pipeline/graph_mapper.py` | modified | Flat theme props on `Event`; scene-theme rollup props on `Scene`; emits `THEMATIC_LINK` edges (C1, C3, D1) |
| `cekg_pipeline/pipeline.py` | modified | New stage 0 alias seeding; `build_jsonld(scenes=...)` (C2) |
| `cekg_pipeline/theme_graph.py` | **new** | Builds Event→Event `THEMATIC_LINK` edges (D1) |

---

## 7. How to Verify

1. Clear stale checkpoints so the new prompts run on linking + theme stages:
   ```bash
   python main.py "Great Expectations.txt" --clear-checkpoints
   ```
2. Run the pipeline:
   ```bash
   python main.py "Great Expectations.txt" --full
   ```
3. Inspect the new outputs:
   - `neo4j_csv/thematic_links.csv` — all subplot beats (one row per
     `(source_event, target_event, theme)` triple).
   - `neo4j_csv/events.csv` — per-theme involvement / role / confidence as
     flat columns instead of a JSON blob.
   - `neo4j_csv/scenes.csv` — `theme_<T>_*` rollup columns.
4. Load into Neo4j via `ge_import.txt` (now contains a
   `CREATE INDEX … FOR ()-[r:THEMATIC_LINK]-() ON (r.theme)` line for
   subplot-by-theme queries) and run the Cypher patterns in §5.

---

## 8. Open Items / Future Work

- The alias dictionary is currently seeded only for *Great Expectations*.
  Extending to other works in `novels/` is a 30-minute task — add canonical
  → aliases entries to `KNOWN_WORK_ALIASES` in
  `cekg_pipeline/coreference_resolver.py`.
- Theme over-tagging is reduced by the prompt + the
  `confidence<0.4 → latent` demotion, but a quantitative precision
  evaluation against a hand-labelled set is the right next step (the
  reviewer's "relation label precision" question in §B).
- The thematic edge layer currently uses a fixed `confidence = sqrt(s·t)`
  combiner. If a particular subplot path matters analytically (e.g. the
  WEALTH spine through Magwitch) we can add a path-aggregation function on
  top.
