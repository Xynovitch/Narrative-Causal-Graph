# Proposal — Additional Subplot Detector

**Date:** 2026-04-28
**Author:** CEKG pipeline maintenance
**Status:** Draft for review (not implemented)

---

## 1. Motivation

The current pipeline already exposes one notion of a subplot:

> **Theme-projected subplots** (from `cekg_pipeline/theme_graph.py`) — for each
> theme T ∈ {POWER, WEALTH, KINSHIP, JUSTICE, KNOWLEDGE}, the connected
> component induced by `Event -[:THEMATIC_LINK {theme: T}]-> Event` is
> treated as the subplot for T.

This works well for the five canonical themes — for example, the *Hidden
Wealth / False Expectations* arc in *Great Expectations* falls cleanly out
of the WEALTH component once Magwitch's reveal is annotated as `revealing`.
But theme-projected subplots have three structural limitations:

1. **Theme-bound.** Any subplot that doesn't align with the five themes is
   invisible. *The Age of Innocence*'s Newland↔Ellen forbidden-love arc, for
   instance, is most naturally a KINSHIP+POWER hybrid, but neither theme
   captures it cleanly on its own — the connected components fragment.
2. **Annotation-bound.** A subplot only emerges if every beat in it was
   annotated as `direct` or `indirect` for the right theme. A single
   missing annotation breaks the chain. The 0326 feedback already flagged
   the related "weak signals over-tag, strong signals get missed" problem.
3. **Character-blind.** Subplots are about characters' transformations as
   often as about themes. Pip's *moral* arc and Pip's *wealth* arc are not
   the same chain of events; the current detector returns the latter only.

### What we want from an additional detector

A detector that produces **subplots not constrained by the five-theme
ontology** and whose definition of cohesion comes from the *event chain
itself* — its agents, motivations, and causal shape — rather than from a
pre-existing label set.

---

## 2. Proposal: a four-signal subplot detector

We propose an **ensemble** detector that produces *candidate subplots* from
four independent signals, then merges them through a single LLM
verification pass. None of the four signals depends on the theme
annotations, so the detector is complementary (not redundant) with the
existing thematic edge layer.

```
┌─────────────────────────┐
│ Signal A:               │  ── candidate ──┐
│ Character-Arc clustering│    subplots     │
└─────────────────────────┘                 │
┌─────────────────────────┐                 │
│ Signal B:               │  ── candidate ──┤
│ Why-Factor coupling     │    subplots     │   ┌────────────────┐
└─────────────────────────┘                 ├──>│ Merge + LLM     │── final subplots
┌─────────────────────────┐                 │   │ verification     │
│ Signal C:               │  ── candidate ──┤   └────────────────┘
│ Embedding topic cluster │    subplots     │
└─────────────────────────┘                 │
┌─────────────────────────┐                 │
│ Signal D:               │  ── candidate ──┘
│ Causal-graph motifs     │
└─────────────────────────┘
```

Each signal is cheap, runs purely on already-extracted CEKG output, and
produces a *ranked list* of candidate subplots. The merge step deduplicates
by event-set Jaccard similarity, and the LLM step labels each surviving
candidate with a name, a one-sentence summary, and a confidence score.

### Output schema

A new artefact, `subplots.json`, alongside the existing exports:

```json
{
  "subplots": [
    {
      "id": "subplot/ge_pip_identity",
      "name": "Pip's identity collapse and reinterpretation",
      "summary": "Pip's social ascent driven by a hidden source collapses when Magwitch reveals the true benefactor.",
      "events": ["event/a30e9de2", "event/82dee4b0", ..., "event/94195f61"],
      "central_agents": ["Philip Pirrip", "Abel Magwitch"],
      "themes": ["WEALTH", "KINSHIP", "JUSTICE"],
      "detector_signals": {"character_arc": 0.92, "why_factor": 0.71,
                           "topic_cluster": 0.84, "causal_motif": 0.66},
      "llm_verification": {"valid": true, "confidence": 0.88}
    },
    ...
  ]
}
```

Crucially, an event can appear in *multiple* subplots — the detector is not
a partition. This matches how literary scholars actually talk about
overlapping arcs.

---

## 3. The four signals

### Signal A — Character-arc clustering

**Hypothesis.** A character-arc subplot is a temporally-ordered chain of
events that (a) all involve a specific agent, and (b) cluster into 2–4
distinct *phases* in the event embedding space, the way a transformation
arc clusters into "before / crisis / after".

**Algorithm.**
1. For each canonical agent A with ≥ 20 events (post-resolver
   normalisation):
2. Take the agent's events ordered by `(chapter, sequence)`.
3. Embed `raw_description` with the existing `all-MiniLM-L6-v2`.
4. Run **changepoint detection** on the rolling-mean embedding to find
   2–4 phase boundaries.
5. The events between the first and last detected boundary form the
   candidate subplot. Confidence is the magnitude of the embedding
   distance between adjacent phases (a flat agent → low confidence, a
   transforming agent → high).

**Sample, expected output on `/novels`.**
- *Great Expectations* — Pip: 4-phase arc (marsh boy → genteel apprentice
  → London gentleman → reformed Pip), boundaries near chapters 1, 8, 18,
  39 — matches the seven sub-stages the 0326 reviewer enumerated by hand.
- *Arrowsmith* — Martin Arrowsmith: 3-phase arc (idealist student →
  compromised industrial researcher → St. Hubert plague crucible).
- *All Quiet on the Western Front* — Paul Bäumer: 2-phase arc (naïve
  soldier → numbed survivor) with a sharp boundary mid-text, mirroring
  the famous "we are forlorn like children" passage.
- *The Sheik* — Diana Mayo: 2-phase arc (independent traveller →
  transformed captive), boundary at the abduction.

Existing infra reused: `sentence-transformers` (already a dep), the
`coreference_resolver`'s alias dictionary (so Pip / the boy / Philip Pirrip
collapse).

### Signal B — Why-Factor coupling

**Hypothesis.** A subplot is a chain of events whose `why_factors` evolve
*coherently* — a single dominant motive that is set up, escalated, and
either fulfilled or thwarted.

**Algorithm.**
1. Build a frequency table of `why_factor` strings across events.
2. For each high-frequency factor F, take all events whose `why_factors`
   include F or a near-synonym (cosine similarity ≥ 0.7 in MiniLM space).
3. Order temporally; require ≥ 5 events; require a non-trivial
   first-half / last-half polarity flip *or* a confidence rise to count
   as a complete arc (the "setup → resolution" shape).
4. Confidence = (events covered) / (total events involving central
   agent), clipped to [0, 1].

**Sample, expected output.**
- *Great Expectations* — `why_factor: "shame"` chains Pip's hand episode,
  Estella's contempt, the gentleman desire, the rejection of Joe — a
  classic shame-driven subplot orthogonal to the WEALTH chain.
- *The Age of Innocence* — `why_factor: "social obligation"` ties
  Newland's engagement, May's family pressure, Ellen's exclusion, and
  Newland's late renunciation into a single arc; this is the subplot
  that the THEMATIC_LINK approach fragments across KINSHIP and POWER.
- *Elmer Gantry* — `why_factor: "ambition"` produces the clean
  rise-and-fall arc of a hypocrisy-driven preacher.

### Signal C — Embedding topic cluster

**Hypothesis.** Some subplots are best detected by *what they are about*
in latent semantic space, irrespective of named agents or themes.

**Algorithm.**
1. Embed every event's `raw_description` (already done elsewhere in the
   pipeline; reuse the cached vectors).
2. Run **HDBSCAN** with `min_cluster_size = 8`, `metric = 'cosine'`.
3. For each cluster, fit the events into temporal order; if the cluster
   spans ≥ 3 chapters, treat it as a candidate subplot.
4. Cluster name = top-3 TF-IDF tokens from the cluster's text. The LLM
   verification step renames it to something readable.

**Why HDBSCAN and not k-means.** Subplot count is unknown ahead of time,
and HDBSCAN tolerates noise (events that don't belong to any subplot),
which is the more honest reading of a novel.

**Sample, expected output.**
- *The Greene Murder Case* — a "crime-scene investigation" cluster
  separates from a "family inheritance dispute" cluster cleanly; both
  are subplots that the five-theme ontology (no JUSTICE-as-investigation
  refinement) couldn't distinguish.
- *Show Boat* — a "river / showboat performance" cluster forms a setting
  subplot independent of the characters' romantic arcs.

### Signal D — Causal-graph motifs

**Hypothesis.** A subplot has a recognisable causal *shape*. The
quintessential narrative arc is `INCITING_CAUSE → ESCALATES* →
CAUSES_REVERSAL → RESOLVES`. Subgraph motif matching on the existing
causal graph finds these directly.

**Algorithm.**
1. Build the directed graph of `CausalLink` edges, ignoring the
   THEMATIC_LINK layer.
2. Look for paths of length 4–15 that match the regular expression of
   `edge_supertype` values:
   - `^CAUSAL_PRODUCTION (CAUSAL_PRODUCTION|EMOTIONAL_DRIVE|NARRATIVE_ESCALATION)+ NARRATIVE_RESOLUTION$`
3. Each match is a candidate subplot; rank by total edge confidence.

This is a structural — almost grammatical — definition of a subplot.
The `edge_supertype` mapping already exists in
`cekg_pipeline/theme_annotation.py::FINE_TO_SUPERTYPE`.

**Sample, expected output.**
- *Great Expectations* — the canonical arc Pip helps Magwitch
  (CAUSAL_PRODUCTION) → Pip ashamed (EMOTIONAL_DRIVE) → Pip seeks gentility
  (NARRATIVE_ESCALATION × N) → Magwitch reveal (REVELATION_EPISTEMIC,
  promoted to NARRATIVE_RESOLUTION) → Pip reinterpretation
  (NARRATIVE_RESOLUTION). This *exact* path was the example the 0326
  reviewer drew by hand.
- *Cimarron* — the "Yancey vanishes / returns / dies" structural arc
  matches the regex despite spanning a 30-year diegetic period.

### Merge + LLM verification

After the four detectors run, candidate subplots are merged:
1. Compute Jaccard overlap on event sets between every pair of
   candidates.
2. Cluster candidates with overlap ≥ 0.5 — same subplot, different
   detector views.
3. For each merged cluster, send the union of events (or a
   representative 30-event sample) to one LLM call:
   ```
   Given these events, are they a coherent subplot? If yes, give a name
   (≤ 8 words), a one-sentence summary, and rate coherence ∈ [0, 1].
   If no, return rejected: true.
   ```
4. Only subplots with `llm_verification.confidence ≥ 0.6` survive.

Cost estimate: one LLM call per surviving candidate, ~50 candidates per
novel → ~$0.10 per novel at gpt-4o-mini, or ~$1 at gpt-5.

---

## 4. Why this is complementary, not redundant

The existing `THEMATIC_LINK` layer answers *"which events participate in
theme T?"*. The proposed detector answers *"what are the discrete
narrative arcs in this novel, regardless of theme?"*. They overlap but
neither subsumes the other, illustrated below.

| Subplot example | Caught by THEMATIC_LINK? | Caught by proposed detector? | Detector signal |
|---|---|---|---|
| Pip's WEALTH arc (Great Expectations) | ✓ (WEALTH component) | ✓ | A + B + D |
| Pip's *shame* arc (orthogonal to WEALTH) | ✗ (no shame theme) | ✓ | B |
| Newland↔Ellen forbidden love | ✗ (KINSHIP + POWER fragment) | ✓ | A + B |
| Diana's captivity transformation (The Sheik) | partial (POWER only) | ✓ | A + D |
| Greene murder investigation cluster | ✗ (no investigation theme) | ✓ | C |
| Paul's wartime disillusionment | partial (JUSTICE only) | ✓ | A + C |
| Magnificent Obsession's medical-redemption arc | ✗ | ✓ | B + D |

The proposed detector therefore expands the *scope* of subplot detection
(by adding character-arc and topic-cluster modes), while keeping the
existing five-theme analysis as the canonical "structural reading".

---

## 5. Integration plan

**Pipeline.** A new optional stage between theme annotation and export:

```
            existing                              new
  theme_annotation ──> exports
                    └─> subplot_detector ──> subplots.json
                                          └─> Subplot nodes in CSV/Cypher
                                          └─> Event -[:IN_SUBPLOT]-> Subplot
```

The detector reads `events`, `causal_links`, and the resolver's canonical
agent map. It writes `subplots.json` and a single new edge type:

```
Event -[:IN_SUBPLOT {role: "setup" | "escalation" | "reversal" | "resolution",
                     position: int,           // 0-indexed within the subplot
                     coherence: float}]-> Subplot
```

Subplot nodes carry the LLM-verified `name`, `summary`, `central_agents`,
`themes` (the five-theme overlap), and per-detector signal scores.

**Code.** New file `cekg_pipeline/subplot_detector.py` (~600 lines)
implementing the four detectors plus the merge step. Reuses existing
embedding model, ontology, and resolver.

**CLI.** New flag `--enable-subplot-detection` (off by default; on under
`--full`). Adds approximately 5 minutes per novel (mostly HDBSCAN +
verification calls).

---

## 6. Validation strategy

A subplot detector is hard to evaluate without ground-truth annotations;
two complementary validations:

1. **Sparknotes-style chapter summaries.** For each of the 13 novels in
   `/novels/`, scrape the public chapter summary and treat the named
   subplots in those summaries as a recall set. Detector recall =
   fraction of named-subplot events that appear in *some* detected
   subplot.
2. **Reviewer alignment.** The 0326 reviewer hand-annotated a
   *Great Expectations* WEALTH subplot with seven beats. We measure the
   detector's overlap (Jaccard ≥ 0.7 on event IDs) with that
   hand-annotation as a single high-confidence anchor. A passing
   detector recovers it from at least two of the four signals.

Both can be automated and re-run when the detector is tuned.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| HDBSCAN is non-deterministic across runs | Pin `random_state` (HDBSCAN doesn't have one — pin the input order instead, sort events by id) |
| Why-factor strings are too noisy to cluster | Pre-normalise with the existing embedding similarity step before frequency counting |
| LLM verification is the cost bottleneck | Cap candidate count at 30 per novel; reuse the BoundedCache |
| Character-arc detection over-fits to protagonists | Require minimum 2-phase change AND minimum 20-event presence — secondary characters with flat trajectories naturally drop out |
| Subplots overlap so much the UI becomes a mess | Surface a "primary subplot" attribute on each event = the subplot whose `coherence × verification_confidence` is highest; secondary subplots remain queryable but visually de-emphasised |

---

## 8. Decision request

Two go/no-go calls:

1. Implement all four signals or start with **A + D** only? Signals A and
   D are the highest-confidence pair (transformation + structural shape)
   and account for ~70% of the subplots in the table in §4. B and C
   broaden coverage but add cost.
2. Land the detector under `--enable-subplot-detection` (opt-in) or
   include it in `--full`? Recommendation: opt-in for one release, then
   promote to `--full` after the validation in §6 hits an agreed
   threshold (e.g. ≥ 0.7 recall on the Sparknotes set).

---

## 9. Appendix — Worked example, *The Age of Innocence*

| Signal | Output |
|---|---|
| A — character arc on Newland Archer | 3-phase arc (engaged conformist → torn lover → resigned husband), boundaries near the engagement announcement, the Skuytercliff weekend, and the post-Ellen final chapter |
| B — why-factor "social obligation" | Setup-pay-off arc spanning 28 events, polarity flip at the Mingott family conclave |
| C — topic cluster | "letters and concealment" cluster (not aligned with any of the five themes) emerges as a distinct subplot |
| D — causal motif | The path Newland-meets-Ellen → escalating private encounters → reversal at the Mrs Mingott telegram → resolution at the Paris-bookshop epilogue is one of three matches that survive the regex |

Merge step combines A + B + D into a single subplot
*"Newland Archer's renunciation arc"*; signal C surfaces independently as
*"Letters as the novel's shadow communication channel"* — distinct, both
valid.

The five-theme `THEMATIC_LINK` layer would have produced *neither* of
these as a single connected component.
