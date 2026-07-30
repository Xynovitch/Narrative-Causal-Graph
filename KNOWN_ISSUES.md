# Known Issues

Open defects in the CEKG pipeline, with measured impact and located fixes. Line numbers refer to `main` as of 2026-05-28.

This list exists so anyone reading the code or the published graphs knows what to trust. Each entry states what it invalidates and — just as importantly — what it does not. Two earlier defects that *did* invalidate results have already been fixed; they are recorded at the bottom for provenance.

**Canonical dataset:** `docs/web/data/gpt_oss_120b_full.json` (gpt-oss-120b via local vLLM, 2026-05-28). Statistics below are measured against it unless stated otherwise.

---

## 1. Event ontology is truncated to its first 20 types

**Severity:** high — invalidates event-type distribution
**Location:** `cekg_pipeline/llm_service.py:267`

```python
ont_str = ", ".join(event_ontology[:20])
```

The extraction prompt offers the model only the first 20 entries of the event ontology. `schema.json` is alphabetically sorted, so the reachable vocabulary is `ACKNOWLEDGMENT` through `CONTEMPLATION`.

**Measured impact.** 22 distinct event types appear across 5,889 events — those 20, plus `PHYSICAL_ACTION` and `PHYSICAL_MOVEMENT` entering through the `validate_event_type` fallback in `pipeline.py`. Seventy-five of the 97 defined, theory-tagged types are unreachable, including `INCITING_EVENT`, `MORAL_CONFLICT`, `SELF_DISCOVERY`, `OPPONENT_CONFRONTATION` and `TRANSFORMATION`.

**What this invalidates:** any claim about the distribution of event types, and any claim that the McKee/Truby event ontology is exercised.
**What it does not:** the causal graph. Relation typing passes the full ontology (see `integrated_semantic.py`), and 92 of 100 relation types are in use.

**Fix.** Pass the full ontology, or select by relevance rather than list order. If prompt length is the concern, the honest version is a two-stage classify (extract free-form, then type against the full vocabulary) rather than a silent slice. Requires a re-run to take effect.

**Related truncations of the same class**, currently latent:
- `llm_service.py:318` — `relation_ontology[:15]` in `assess_pairs_bulk`. Only reached via the `optimized_linking` fallback path; the primary path in `integrated_semantic.py` passes the full ontology.
- `llm_service.py:375` — `agent_type_names[:20]` in `classify_agent_type`. Affects runs using `--enable-agent-classification`.

---

## 2. Causal assessment is blind to narrative distance

**Severity:** high — affects edge precision, not edge existence
**Location:** `cekg_pipeline/integrated_semantic.py:92-102`

Each candidate pair reaches the model as two descriptions truncated to 80 characters and nothing else. The `CEKEvent` objects carry `chapter`, `sequence`, `action_type`, `actors`, `patients`, `why_factors` and `location_context`; none are passed. The model therefore cannot know whether it is judging adjacent events or events fifty chapters apart.

**Measured impact.** Confidence is effectively flat with respect to chapter distance:

| chapter span | edges | mean confidence | generic relation share |
|---|---|---|---|
| same chapter | 9,453 | 0.670 | 63.9% |
| 1–4 | 1,838 | 0.656 | 60.6% |
| 5–19 | 4,223 | 0.649 | 59.5% |
| 20+ | 5,360 | 0.649 | 59.4% |

Two points of confidence across 58 chapters of separation. 5,360 edges (26% of the graph) assert causation at spans of 20+ chapters with the same confidence as adjacent-event causation.

Genuine long-range links exist — e.g. `ch5→ch47: "convict's warning triggered narrator's recollection of the chase"`, a real retrospective relation. So do lexical coincidences — e.g. `ch5→ch56: "convicts' distressed state prompts Judge to address them"`, 51 chapters apart, joined only by the word "convict".

**What this invalidates:** precision of long-span edges. Treat edges with `sequence_distance` in the top quartile as candidates rather than findings.
**What it does not:** local causal structure, or the existence of long-range links as a class.

**Fix.** Include `chapter` and sequence distance in the pair line, and raise the 80-character truncation. Both are small changes to the prompt construction in `assess_pairs_causal`. Interacts with issue 3 — better candidates and better context should land together.

---

## 3. BM25 candidate pool is unthresholded

**Severity:** medium — inflates candidate volume with low-quality pairs
**Location:** `cekg_pipeline/dynamic_context.py:244-246`, constant at `:47`

```python
top_indices = np.argsort(scores)[-top_k:]
for j in top_indices:
    if scores[j] <= 0:
        continue
```

Every event contributes its top-10 BM25 neighbours, gated only by `score > 0`. BM25 scores are essentially never ≤ 0 given any lexical overlap, so the gate never fires. BM25 is the only candidate pool that is both linear in N × k and effectively ungated — every other pool is either bounded by structure (adjacency, entity chains) or by a similarity threshold plus a cap (scene ≥ 0.75 with 30/scene; long-shot ≥ 0.50 with window restriction).

**Measured impact** (BM25Okapi reproduced over the event descriptions):

- **100% of the N × 10 budget is accepted** — every event keeps all ten neighbours.
- Median selected pair scores **27%** of the querying event's own self-score; 63% fall below 30%.
- Rank-1 median 19.7 vs rank-10 median 13.8 — a shallow gradient, meaning there is no relevance cliff and the top-10 cutoff is arbitrary.
- Descriptions average 15.2 tokens with no stopword removal or lemmatization. The shortest quartile scores *higher* relative to self (43.6% vs 35.4%), the signature of matching on function words.

**Fix.** Add a relative-score threshold: at 0.25 of self-score, 60% of pairs are retained; at 0.35, 21%. Alternatively replace top-k with top-p (keep neighbours within X% of rank-1), which adapts to events that genuinely have many strong matches. Stopword removal and lemmatization on the BM25 corpus would help independently.

---

## 4. `CausalLink.weight` is always 0.0

**Severity:** low — dead field
**Location:** `cekg_pipeline/integrated_semantic.py:212`

```python
weight=float(c_res.get("weight", 0)),
```

The assessment prompt never asks for a weight, so the key is never present and the default is always taken. All 20,874 causal edges in the current dataset have `weight = 0.0`. The field is exported to CSV, JSON-LD and Cypher.

**Fix.** Either request it in the prompt with a stated meaning distinct from `confidence`, or remove it from the schema and exporters. Removing is probably right — `confidence` already carries the model's certainty, and a second unexplained scalar invites misreading.

---

## 5. `Place` nodes and `HOSTS` edges are never produced

**Severity:** low — schema/output mismatch
**Location:** `cekg_pipeline/pipeline.py` (`_parse_event_json_data`), `cekg_pipeline/exporters.py`

`schema.json` defines 48 `PlaceType` entries and the exporters emit `places.csv` and `hosts.csv`, but nothing in the pipeline ever constructs an `EventProducesEntity` with `entity_type="place"`. Only `actor`, `patient` and `whyfactor` are emitted (6,342 / 2,223 / 6,777 in the current run). Both CSVs are empty. `Scene.place_type` and `Scene.time_type` are likewise always `None`.

Events do carry `location_context` as a string, so the information is present — it is simply never promoted to a node.

**Fix.** Promote `location_context` to a `Place` entity with `HOSTS` edges, and classify against `PlaceTypeDictionary` the way agents are classified against `AgentTypeDictionary`. Until then, README and schema documentation should not imply `Place` is populated.

---

## 6. A superseded dataset is still published

**Severity:** medium — a reader can select broken data without knowing
**Location:** `docs/web/data/great_expectations_gpt5.json`, listed in `manifest.json`

The GPT-5 run of 2026-05-07 predates both fixes recorded below. Verified against the published copy: 8,625 events, 13,991 causal edges, **0% cross-chapter (max chapter span 0)**, 33 distinct relation types in a near-uniform distribution. It is selectable in the web explorer's dataset dropdown alongside the current run.

**Fix.** Remove it from `docs/web/data/` and `manifest.json`, or label it in the UI as superseded. Keeping it as a labelled contrast is defensible; keeping it silently selectable is not.

---

## Fixed — recorded for provenance

Both of these invalidated results and are closed. Any output generated before 2026-05-27 should be discarded.

**LLM cache-key collisions** (fixed in `d4b8a9f`, 2026-05-20). The assessment cache key hashed only `(prefix, batch-size, ontology-prefix)` rather than prompt content, so the first 50-pair batch's response was replayed for every subsequent batch. Symptom: a near-uniform relation-type distribution (502 / 501 / 499 / 498 / 498) that looked plausible in aggregate. The current run shows 92 relation types with a natural long tail.

**Import guard silencing every cross-chapter signal** (fixed in `9725d71`, 2026-05-27). `get_bm25_pairs` was gated on `not _BM25_AVAILABLE or not _EMBED_AVAILABLE`, and `sentence-transformers` was failing to import (Pillow 9.0.1 lacking `PIL.Image.Resampling`). A single import failure therefore silenced BM25, the scene pool and the long-shot pool simultaneously, leaving only chapter-local candidate pools. Symptom: **zero cross-chapter causal links out of 13,991.** The current run is 54.7% cross-chapter, longest span 58 chapters.

Both failures were silent: counts looked normal, the pipeline reported success, and no count-based regression check would have caught either. What surfaced them was rendering the graph and seeing disconnected islands instead of a connected causal web. A connectivity check — component count, largest-component fraction, cross-chapter edge share — is now the cheapest guard against this class of failure and should be part of any regression suite.

---

## Not defects, but worth knowing

**Acyclicity is guaranteed by construction, not discovered.** `DAGValidator.add_edge` requires `cause.sequence < effect.sequence`, which makes cycles impossible; the DFS cycle check and Kahn's-algorithm validation in `utils.py` are redundant guards. "DAG valid: yes" in the output is true but weakly informative.

**`sequence` is discourse order, not story order.** It is assigned by an incrementing counter as text is parsed. Combined with the constraint above, this means a cause narrated *after* its effect cannot be represented — a real limitation for retrospective narratives such as *Great Expectations*, where the central causal fact is disclosed in chapter 39.

**The event ontology is theory-seeded but text-expanded.** `scripts/generate_ontology.py` seeds an LLM with McKee and Truby beat names and abstracts categories from 13 novels; the `@McKee` / `@Truby` tags on the expanded types were assigned afterward. Types like `INTERRUPTED_MEAL` are artifacts of that process, not claims about either theory.

**No evaluation against ground truth.** Testing is a regression check — run on *Great Expectations*, compare counts to a documented baseline. That catches breakage, not correctness. There is no precision/recall measurement for extracted events or causal edges.
