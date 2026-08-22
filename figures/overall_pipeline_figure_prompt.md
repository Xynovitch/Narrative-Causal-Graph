# Overall Pipeline Figure Prompt

Reference style: use the overall project flowchart style from `C:\Users\6seve\Codelib-severin\1_Research\CuraView\论文\figures\01_系统总流程图.pdf`. That reference is the CuraView total project pipeline figure: a wide landscape architecture diagram with numbered layer labels on the left, a large central staged pipeline, a right-side dashed improvement loop, and a compact bottom legend.

Use this prompt to generate the image:

```text
Create a publication-quality, vector-style architecture diagram for a research paper.

Title: "Narrative Causal Event Knowledge Graph: From Raw Novel Text to Queryable Causal and Thematic Graphs"

Canvas and style:
- Wide landscape layout, approximately 16:9 or 2:1 aspect ratio, white background.
- Follow the visual grammar of the CuraView overall flowchart: numbered layer panels on the far left, color-coded horizontal layers in the center, a dashed feedback/resume loop on the right, and a compact legend along the bottom.
- Use clean academic diagram styling: thin black outlines, rounded rectangles with small radius, crisp arrows, restrained colors, no gradients, no shadows, no 3D effects, no decorative backgrounds.
- Use simple black line icons where useful: book, chapter pages, ontology/schema document, LLM chip, person/agent, map pin, clock, graph nodes, causal arrow, theme tag, database/export, browser graph explorer.
- Make all text readable at paper size. Use short labels inside boxes and avoid long paragraphs.
- Use consistent typography, bold section headers, and enough spacing so arrows and labels never overlap.

Overall layout:
- Left column: four large numbered layer labels, stacked vertically.
- Center: the main CEKG pipeline, flowing downward across layers and left-to-right within each layer.
- Right column: a dashed vertical box named "Checkpoint, Review, and Exploration Loop" with arrows back to earlier stages.
- Bottom: a legend explaining node types, edge types, theme colors, and arrow styles.

Layer 1 label on the left:
"1 Narrative Text and Ontology Layer"
Subtitle: "(Raw Text + Theory Schema)"

Layer 1 central content:
Use a blue-outlined container with two parallel input groups:

Input group A: "Raw Narrative Text"
- Show a book icon and `.txt novel`.
- Include preprocessing steps:
  - "load_text()"
  - "strip Project Gutenberg boilerplate"
  - "split_chapters()"
  - "fallback paragraph split"
- Output object: "Chapters and chapter text".

Input group B: "Narrative Ontology and Runtime Config"
- Show schema document icon.
- Include:
  - `schema.json`
  - 47 agent types
  - McKee and Truby causal relation types
  - event, place, and time vocabularies
  - 5 structural themes: POWER, WEALTH, KINSHIP, JUSTICE, KNOWLEDGE
  - OpenAI or local vLLM backend
- Output object: "Ontology-constrained prompts and validators".

Draw both outputs downward into Layer 2.

Layer 2 label on the left:
"2 Event Extraction and Context Grounding Layer"
Subtitle: "(CEKEvent + Entity-Event Links)"

Layer 2 central content:
Use a green-outlined container with a left-to-right pipeline:
1. "Chunked Chapter Processing"
   - wtpsplit sentence segmentation
   - overlapping chunks
   - async LLM calls with semaphore
   - retry and cache
2. "LLM Event Extraction"
   - extract who, did what, where, when, why
   - source quote and confidence
   - event type under McKee/Truby ontology
3. "Coreference Resolution"
   - canonical character names
   - aliases, pronouns, descriptors
   - learned co-occurrence
4. "CEKEvent Objects"
   - event_id, raw_description, actors, patients, why_factors, place, time, chapter, sequence
5. "Context Propagation"
   - propagate location and time
   - propagate actors and motivations
   - create ACTS_IN, AFFECTED_IN, MOTIVATES, HOSTS links

Show output card:
"Grounded Event Sequence"
Subtitle: "Events plus agent, place, time, and motivation links."

Draw a small checkpoint icon after this layer labeled "checkpoint: extraction + context_propagation".

Layer 3 label on the left:
"3 Causal Graph Construction Layer"
Subtitle: "(Candidate Pairs + LLM Causal Assessment)"

Layer 3 central content:
Use an orange-outlined container with three subtracks that converge:

Subtrack A: Optional narrative structure enrichment
- "Agent Classification"
  - classify characters into ontology roles such as protagonist, antagonist, revelation giver.
- "Scene Grouping"
  - group events by spatial, temporal, and thematic coherence.

Subtrack B: Candidate pair generation
- Large box: "Dynamic Context Candidate Pair Generator"
- Inside it, show six strategy chips:
  - adjacent and local windows
  - scene-internal pairs
  - double sliding long-range pairs
  - embedding similarity
  - BM25 keyword overlap
  - entity-guided pairs
- Add label: "cap by --max-pairs; avoid O(N^2)".
- Also show a smaller fallback box:
  "IntelligentCausalLinker fallback: entity co-occurrence, temporal proximity, chapter transitions, narrative peaks".

Subtrack C: LLM causal assessment and validation
- Box sequence:
  "Candidate event pairs" -> "Counterfactual causal prompt" -> "McKee/Truby relation type" -> "CausalLink"
- In the CausalLink box, list:
  - source_event_id -> target_event_id
  - relation_type
  - mechanism
  - confidence
  - theory
  - edge_supertype
- Add "DAG validation" as a gate before output.

Show output card:
"Directed Causal Event Graph"
Subtitle: "Event nodes connected by typed causal relations."

Draw a small checkpoint icon labeled "checkpoint: scenes + linking".

Layer 4 label on the left:
"4 Thematic Annotation, Export, and Exploration Layer"
Subtitle: "(Theme Chains + Research Artifacts)"

Layer 4 central content:
Use a purple-outlined container with a left-to-right pipeline:
1. "Local Causal Context"
   - event text
   - actors, patients, chapter, scene
   - up to 2 immediate causes
   - up to 2 immediate effects
2. "LLM Theme Annotation"
   - theme participation is local causal mechanism, not broad literary interpretation.
   - 5 themes: POWER, WEALTH, KINSHIP, JUSTICE, KNOWLEDGE.
   - involvement: direct, indirect, latent, none.
   - role: initiating, enabling, constraining, mediating, escalating, resolving, revealing.
3. "Theme-Bridge Rule"
   - if an adjacent causal neighbor has direct theme involvement, upgrade micro-events to indirect mediating involvement.
4. "ThematicLink Construction"
   - deterministic THEmatic edges from shared theme participation.
   - build subplot-like connected components per theme.
5. "Multi-format Export"
   - JSON-LD: `ge_preprocessed.json`
   - Neo4j Cypher: `ge_import.cypher`
   - CSV: events, agents, scenes, causes, thematic_links, follows, acts_in, affected_in, motivates.
6. "Graph Explorer"
   - causal view
   - subplot/theme view
   - agent view
   - node focus view
   - filters by chapter, character, why factor, theme, edge type, confidence

At the bottom of the central pipeline, show the final output in a wide blue-outlined bar:
"Output: Queryable Narrative Causal Event Knowledge Graph"
Subtitle: "Events, agents, scenes, causes, thematic links, exports, and web exploration artifacts."

Right-side dashed loop:
Dashed blue container titled "Checkpoint, Review, and Exploration Loop".
Stack these boxes vertically:
1. "Stage Checkpoints and Resume"
   - text_split, extraction, context_propagation, agent_classification, scenes, linking, theme_annotation.
2. "Cache and Cost Control"
   - LLM response cache
   - dynamic max_tokens
   - max pairs
   - fast/full modes
3. "Graph Explorer Review"
   - inspect causal chains
   - inspect thematic subplots
   - inspect agent-centered paths
   - inspect source quotes
4. "Schema and Prompt Tuning"
   - adjust ontology terms
   - refine causal prompt
   - refine theme prompt
5. "Rerun or Resume Pipeline"
   - resume from last checkpoint
   - regenerate exports and web data
Draw dashed arrows from this loop back to:
- ontology and prompt configuration in Layer 1,
- event extraction and coreference in Layer 2,
- candidate generation and causal assessment in Layer 3,
- theme annotation and export in Layer 4.

Bottom legend:
- Green circle: Event node.
- Gray circle: Agent, Place, Time, or WhyFactor entity.
- Purple rounded box: Scene group.
- Black solid arrow: main data flow.
- Black directed edge: CAUSES relation.
- Gray arrow: FOLLOWS chronological relation.
- Green arrow: ACTS_IN or AFFECTED_IN participant relation.
- Purple dashed edge: THEMATIC_LINK.
- Blue dashed arrow: checkpoint, resume, review, or improvement loop.
- Theme color chips:
  - POWER
  - WEALTH
  - KINSHIP
  - JUSTICE
  - KNOWLEDGE

Important accuracy constraints:
- Do not depict the project as a generic summarizer or a simple knowledge graph extractor.
- The primary node is the narrative event, not a paragraph or document chunk.
- The causal graph is built through candidate pair generation plus LLM counterfactual causal assessment, not all-pairs comparison.
- Thematic links are derived from local causal-context theme annotations and deterministic theme rules, not from generic embedding similarity.
- The web explorer is downstream of CSV/JSON-LD/Cypher exports, not part of the extraction model itself.
- Use the exact terms listed above and keep all text in English.
```
