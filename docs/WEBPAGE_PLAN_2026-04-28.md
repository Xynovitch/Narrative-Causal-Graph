# CEKG Research-Presentation Webpage — Implementation Plan

**Date:** 2026-04-28
**Goal:** A self-contained, queryable webpage that exposes the full CEKG output (events, themes, subplots, scenes, agents, causal links) for a research presentation. Audience can pick which entities/relations to look at, follow subplots, drill into a single event, or compare across themes — without installing Neo4j.

---

## 1. Constraints from the research-presentation context

| Constraint | Implication |
|---|---|
| Must be sharable as a single artifact (URL or zip) | Static HTML + assets, no server runtime required for the demo |
| Audience may include non-technical scholars | UI must speak in narrative-theory terms (events, scenes, subplots, themes), not in graph-DB terms (nodes, properties, Cypher) |
| Has to handle one full novel (~5–10K events, ~25K causal pairs evaluated, ~500 thematic edges) | Use a graph layout that scales (Cosmograph, sigma.js, or Cytoscape with WebGL); load the JSON-LD lazily by chapter where possible |
| Reproducible from `ge_preprocessed.json` and `neo4j_csv/` | One build script that takes those artifacts and produces the static page |
| Should also support multiple novels | Build once per novel; presentation shell switches between them |

---

## 2. Stack decision

**Static HTML + vanilla JS + Cosmograph (WebGL graph) + Lunr (full-text search) + a small index built at build time.**

Rationale:
- **No server.** A single `dist/` directory works on GitHub Pages, S3, file://, or zipped on a USB stick.
- **Cosmograph** handles 100K+ nodes/edges with smooth pan/zoom in the browser; Cytoscape struggles past ~20K. Sigma is fine but Cosmograph's WebGL layer is dramatically faster for the audience experience.
- **Lunr** lets us full-text-search event descriptions and source quotes without a backend.
- **Vanilla JS** keeps the build trivial — no React/Vite churn for a research deliverable that won't be maintained long-term.
- The existing `graph_explorer.html` already takes this shape; the plan extends it rather than rewriting.

Alternatives considered and rejected:
- *Neo4j Bloom / Browser:* requires a running Neo4j; not portable.
- *Streamlit / Dash:* needs a Python process; not shareable as an artifact.
- *React + ReactFlow:* over-engineered for a static research demo and slower at scale.
- *Observable notebook:* good for individual charts but awkward as a multi-panel app.

---

## 3. UI layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ TOP BAR                                                              │
│  [Novel ▼]  [Search: "Pip + WEALTH" ...]  events: N  links: M  ...   │
├──────────────────────┬────────────────────────────┬─────────────────┤
│ FILTER PANEL (left)  │ GRAPH CANVAS (centre)      │ DETAIL PANEL    │
│  ── Themes ────────  │                            │ (right)         │
│   ☑ POWER            │   (Cosmograph WebGL)       │                 │
│   ☑ WEALTH           │                            │  Event: e/abc1  │
│   ☐ KINSHIP          │   nodes coloured by:       │  Chapter 39     │
│   ☐ JUSTICE          │     theme | scene | agent  │  "Pip, dear     │
│   ☑ KNOWLEDGE        │   edges shown:             │   boy, I've     │
│  ── Subplot mode ──  │     ☑ THEMATIC_LINK        │   made a        │
│   ◯ All themes       │     ☑ CAUSES               │   gentleman     │
│   ◉ Single subplot   │     ☐ FOLLOWS              │   on you!"      │
│       ┃              │     ☑ ACTS_IN              │                 │
│       └ WEALTH ▼     │                            │  Themes:        │
│  ── Scenes ────────  │                            │   WEALTH direct │
│   [chapter range]    │                            │     resolving   │
│   1 ────────── 59    │                            │     conf 0.92   │
│  ── Agents ────────  │                            │                 │
│   ☑ Pip              │                            │  Caused by:     │
│   ☑ Magwitch         │                            │   • event/abc0  │
│   ☐ Estella          │                            │     (REVEALS)   │
│   ...                │                            │                 │
│  ── Causal types ──  │                            │  Subplot:       │
│   …                  │                            │   WEALTH chain  │
│                      │                            │   (12 events)   │
└──────────────────────┴────────────────────────────┴─────────────────┘
                ┌──────────────────────────────────────────┐
                │ TIMELINE PANEL (bottom, collapsible)     │
                │  events laid out by chapter × sequence,  │
                │  shaded by theme. Subplot = colored line │
                └──────────────────────────────────────────┘
```

### Filter panel: every CEKG entity is a facet

| Facet | Source field | Multi-select? | Default |
|---|---|---|---|
| Novel | per-build | one-of | first |
| Theme involvement | `theme_<T>_involvement` ∈ {direct, indirect} | yes | all |
| Theme role | `theme_<T>_role` | yes | all |
| Theme min confidence | `theme_<T>_confidence ≥ x` | slider | 0.4 |
| Subplot | connected components on `THEMATIC_LINK` per theme | one-of *or* "all" | all |
| Scene | `scene.id`, `scene.theme` | yes | all |
| Chapter range | `event.chapter` | range slider | full |
| Agent | canonical name (post-resolver) | yes | all |
| Causal relation type | `CausalLink.relation_type` | yes | all |
| Causal supertype | `CausalLink.edge_supertype` | yes | all |
| Theory | `theory ∈ {@McKee, @Truby}` | yes | both |

A filter expression is the **AND** of all facet constraints, and is applied to **events**. Edges are then shown if *both* endpoints survive the event filter and the edge's own type is enabled.

### Five view modes

1. **Causal view** — show events + CAUSES edges. The default narrative graph.
2. **Subplot view** — show events that participate in the selected theme + only `THEMATIC_LINK` edges with `theme = T`. Renders the connected component as a single subplot.
3. **Scene view** — events grouped into scene "boxes"; INCLUDES edges hidden, FOLLOWS edges shown as the scene's spine.
4. **Agent view** — pick one agent (e.g. Pip), see only events they ACT_IN/AFFECTED_IN plus all edges among those events.
5. **Comparison view** — pick two themes; render their two subplots side by side, draw cross-edges where the same event participates in both.

### Detail panel (right side, on click)

For an **Event** click, show:
- Source quote (truncated 300 chars)
- Chapter, sequence, scene
- Per-theme involvement / role / confidence (as a small 5-row table, color-coded)
- Actors / patients / why-factors as chips → click jumps to agent view
- "Caused by" list and "Causes" list with relation type + mechanism
- "Subplot membership" — one chip per theme this event participates in; click jumps to subplot view

For a **THEMATIC_LINK edge** click:
- Theme, source role → target role, confidence
- `via` provenance (causal:<RELATION> or scene_spine:<scene_id>)
- "Walk this subplot" button — pans/zooms to fit the connected component

For a **Scene** click:
- Title, chapter, location, time, summary
- Theme-rollup bars (`theme_<T>_direct_count`, `_indirect_count`, `_avg_confidence`)
- Top participants
- Member events list

### Timeline panel

Bottom strip:
- X axis: chapter (or sequence within chapter)
- Y axis: theme rows (POWER / WEALTH / KINSHIP / JUSTICE / KNOWLEDGE)
- Each event = a dot, opacity = confidence, color = role
- Subplot beats = lines connecting dots within the same theme row
- Brushing the timeline filters the main graph

This is the panel that makes the "Hidden Wealth / False Expectations" subplot visible *as a single shape* across the novel — directly addressing the example in the 0326 feedback.

---

## 4. Data layer

### Build artifacts (per novel)

Generated by a `build_webpage.py` that consumes pipeline output:

```
dist/
├── index.html
├── app.js           (UI logic, ~600 lines)
├── styles.css
├── lib/
│   ├── cosmograph.min.js
│   └── lunr.min.js
└── data/
    ├── manifest.json                     // novel metadata
    ├── ge/                               // one folder per novel
    │   ├── nodes.json                    // events + scenes + agents
    │   ├── edges.causes.json
    │   ├── edges.thematic.json           // {theme, source_role, ...}
    │   ├── edges.acts_in.json
    │   ├── subplots.json                 // pre-computed connected components
    │   ├── scene_index.json              // scene -> [event_ids]
    │   ├── agent_index.json              // canonical name -> [event_ids]
    │   ├── search_index.lunr.json
    │   └── meta.json                     // counts, ranges, vocabularies
    └── arrowsmith/
        └── …
```

Pre-computed indexes (built once):
- `subplots.json` — for each theme, the connected components on `THEMATIC_LINK` filtered by `theme=T`. Each subplot has an ordered event list and a confidence summary.
- `agent_index.json` — pre-resolved canonical names (the resolver alias dictionary already collapses Pip/Philip Pirrip/Magwitch etc, so this is direct).
- `scene_index.json` — `scene_id → ordered list of event_ids` from `Scene.included_event_ids`.
- `search_index.lunr.json` — lunr index over `raw_description` and `source_quote`.
- `meta.json` — chapter range, character list, theme distribution, all dropdown vocabularies.

These pre-computed indexes are what make filters O(1) lookups instead of O(N) scans on every keystroke.

### Source-of-truth mapping

| Webpage data file | Built from | Pipeline file |
|---|---|---|
| `nodes.json` | `events.csv` + `scenes.csv` + `agents.csv` | `cekg_pipeline/exporters.py::export_csv` |
| `edges.causes.json` | `causes.csv` | same |
| `edges.thematic.json` | `thematic_links.csv` | same |
| `edges.acts_in.json` | `acts_in.csv` + `affected_in.csv` | same |
| `subplots.json` | computed in build script | union-find over `thematic_links` per theme |
| `search_index.lunr.json` | events.csv text fields | build script |

The build script reads CSVs directly so it stays in sync with whatever schema the pipeline emits — no second source of truth.

### Loading strategy at runtime

- Load `manifest.json` immediately (~1 KB).
- On novel select, load `meta.json` + `nodes.json` + `subplots.json` (~10–20 MB for *Great Expectations*).
- Load edge files lazily per active view mode (causal vs thematic vs ACTS_IN); a single edge file is at most ~3 MB.
- Search index loaded only when the search box gets focus.
- Total cold-start: <2 s on a research-grade laptop.

---

## 5. Implementation milestones

| # | Milestone | Deliverable | Effort |
|---|---|---|---|
| **M1** | Build script | `build_webpage.py` that reads `neo4j_csv/` + `ge_preprocessed.json` and emits `dist/data/<novel>/*.json` (no UI yet) | 0.5 day |
| **M2** | Static shell | `index.html` with three-panel layout, top bar, novel selector, theme key | 0.5 day |
| **M3** | Causal view | Cosmograph rendering the events + CAUSES edges, click → detail panel | 1 day |
| **M4** | Filter panel | All facets from §3 wired to the graph; AND-of-facets filter expression | 1 day |
| **M5** | Subplot view | Switch to thematic edges; one-of theme selector; connected-component highlight | 1 day |
| **M6** | Scene + agent views | Scene grouping render; agent-centric mode; canonical alias coverage check | 1 day |
| **M7** | Timeline panel | Bottom panel with theme rows, brushing-as-filter, subplot lines | 1 day |
| **M8** | Search | Lunr index, search box wires to "highlight matching events" | 0.5 day |
| **M9** | Multi-novel | Run M1–M8 over a second novel from `/novels/`, novel switcher in top bar | 0.5 day |
| **M10** | Polish for presentation | Color palette tuned for projector, screenshot mode, "story mode" tour for the audience | 1 day |

**Total: ~8 working days.** M1–M5 (4 days) is enough to demo the central claim from the 0326 feedback (the "Hidden Wealth / False Expectations" subplot).

---

## 6. Specific feature → query plan mapping

These are the questions the reviewer in the 0326 feedback explicitly wanted to ask and couldn't:

| Question | UI path | Underlying query |
|---|---|---|
| "Show me Pip's WEALTH-related events" | Filter panel: Agent=Pip + Theme=WEALTH (involvement≥indirect) | `events.filter(e => e.agents.includes(canon('Pip')) && e.theme_WEALTH_involvement !== 'none')` |
| "Show me the WEALTH subplot in story order" | Subplot view + Theme=WEALTH | walk `subplots.WEALTH` ordered by sequence |
| "Compare Pip's role in the WEALTH subplot vs the JUSTICE subplot" | Comparison view: WEALTH × JUSTICE, agent filter Pip | union of two subplots, intersection highlight |
| "Find scenes dominated by KNOWLEDGE" | Scene view, sort scenes by `theme_KNOWLEDGE_direct_count desc` | server-side rank, top-K |
| "Why is event/abc tagged as KINSHIP indirect?" | Detail panel: KINSHIP row → expand → shows `bridge_source`, `bridge_relation`, evidence | render the bridge-rule fields directly |
| "Show me all Magwitch / convict / unknown man events as one trail" | Agent view + Agent=Magwitch (alias dictionary collapses surface forms) | `agent_index['Abel Magwitch']` |
| "Plot units: collapse 'Pip helps Magwitch' to one node" | Scene view → scene-as-node mode | render Scene nodes only, with theme rollups visible |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 10K events too many for the layout to feel responsive | Default-collapse to scene-as-node mode for full-novel views; explode to event-level only when zoomed in or filtered |
| Cosmograph is overkill / single-vendor | Build a thin adapter so we can swap to sigma.js or Cytoscape if needed |
| The 5-theme color palette clashes on a projector | Pick a palette in `oklch` space with ≥ 1.5 perceptual delta between themes; provide a B/W mode for printable handouts |
| Subplot connected components can be huge for KNOWLEDGE (almost every event has it) | Add a confidence threshold to subplot-component computation (default 0.5); show it as a slider |
| The pipeline schema may evolve (e.g. new theme added) | The build script's CSV→JSON step is column-driven; adding a column doesn't break the page, it just shows up as an extra facet |
| Audience has no terminal/tooling | Ship `dist.zip` with `README.md` saying "double-click `index.html`" — works from `file://` |

---

## 8. Pre-presentation checklist (M10)

- [ ] Pre-load *Great Expectations* and one second novel.
- [ ] Pre-bookmark the WEALTH subplot view as the "money shot" — one click from the title slide.
- [ ] Sanity-check Pip alias coverage (resolver dictionary should give Pip ≥ 95% of expected events).
- [ ] Hide the dev console panel for the talk.
- [ ] Confirm runs offline (`file://`).
- [ ] Confirm runs on the venue's projector resolution.
