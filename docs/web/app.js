/* CEKG webpage. Loads a pipeline JSON-LD file and renders an interactive view. */

const THEMES = ["POWER", "WEALTH", "KINSHIP", "JUSTICE", "KNOWLEDGE"];
const THEME_COLOR = {
  POWER: "#e15759", WEALTH: "#f1ce63", KINSHIP: "#59a14f",
  JUSTICE: "#8cd17d", KNOWLEDGE: "#4e79a7",
};
const ACTION_COLOR_DEFAULT = "#7a8aa8";

// Causal edge_supertype -> color, so the graph distinguishes "X enables Y"
// from "X blocks Y" from "X reveals Y" at a glance. Categories come from
// FINE_TO_SUPERTYPE in cekg_pipeline/theme_annotation.py.
const SUPERTYPE_COLOR = {
  CAUSAL_PRODUCTION:    "#4eae5b",  // green: production / enabling
  CAUSAL_CONSTRAINT:    "#d24747",  // red: blocking / constraint
  EMOTIONAL_DRIVE:      "#e98abf",  // pink: emotional cause
  SOCIAL_BOND:          "#e8973c",  // orange: relational tie
  NARRATIVE_ESCALATION: "#e15759",  // red-orange: rising tension
  NARRATIVE_RESOLUTION: "#76b7e0",  // light blue: closure
  REVELATION_EPISTEMIC: "#9b6bd9",  // purple: information surfaces
  MEDIATION_TRANSFER:   "#f1ce63",  // yellow: transfer / delegation
  THEMATIC_CONTRAST:    "#7ec6c2",  // teal: structural contrast
  THEMATIC_EXPLANATION: "#a3b87f",  // muted green: explanatory
};
const SUPERTYPE_DEFAULT = "#5a607a";

// JS-side fallback for relation_type -> edge_supertype, used when the data
// was emitted by an older pipeline run whose FINE_TO_SUPERTYPE map didn't
// cover gpt-5's vocabulary. Mirrors cekg_pipeline/theme_annotation.py.
const RELTYPE_TO_SUPERTYPE = {
  // Production family
  CAUSES: "CAUSAL_PRODUCTION", DIRECT_CAUSE: "CAUSAL_PRODUCTION",
  ENABLES: "CAUSAL_PRODUCTION", FACILITATES: "CAUSAL_PRODUCTION",
  TRIGGERS: "CAUSAL_PRODUCTION", INCITING_CAUSE: "CAUSAL_PRODUCTION",
  EVENT_ENABLES_NEXT: "CAUSAL_PRODUCTION", EVENT_REINFORCEMENT: "CAUSAL_PRODUCTION",
  DESIRE_ALIGNMENT: "CAUSAL_PRODUCTION", NECESSITATES: "CAUSAL_PRODUCTION",
  FULFILLS: "CAUSAL_PRODUCTION", PRECEDES: "CAUSAL_PRODUCTION",
  SCENE_CAUSATION: "CAUSAL_PRODUCTION", SCENE_CHAINING: "CAUSAL_PRODUCTION",
  PLOT_PROPULSION: "CAUSAL_PRODUCTION", STRUCTURAL_DEPENDENCE: "CAUSAL_PRODUCTION",
  CONSEQUENCE_CHAINING: "CAUSAL_PRODUCTION", REINFORCES_GOAL: "CAUSAL_PRODUCTION",
  // Constraint family
  PREVENTS: "CAUSAL_CONSTRAINT", BLOCKS: "CAUSAL_CONSTRAINT", INHIBITS: "CAUSAL_CONSTRAINT",
  COMPLICATES: "CAUSAL_CONSTRAINT", OPPOSES: "CAUSAL_CONSTRAINT",
  DESIRE_OBSTRUCTION: "CAUSAL_CONSTRAINT", DESIRE_COMPETITION: "CAUSAL_CONSTRAINT",
  PHYSICAL_BLOCKAGE: "CAUSAL_CONSTRAINT", INTERRUPTION_OBSTACLE: "CAUSAL_CONSTRAINT",
  MISSION_FAILURE: "CAUSAL_CONSTRAINT", MISSION_ABANDONMENT: "CAUSAL_CONSTRAINT",
  OPPOSITION_PRESSURE: "CAUSAL_CONSTRAINT", PREVENTS_OUTCOME: "CAUSAL_CONSTRAINT",
  RELATIONAL_FRAGMENTATION: "CAUSAL_CONSTRAINT",
  // Emotional drive
  COMPASSION_TRIGGER: "EMOTIONAL_DRIVE", EMOTIONAL_MANIPULATION: "EMOTIONAL_DRIVE",
  EMOTIONAL_DEPENDENCE: "EMOTIONAL_DRIVE", EMOTIONAL_TRIGGER: "EMOTIONAL_DRIVE",
  EMOTIONAL_CONTAGION: "EMOTIONAL_DRIVE", EMOTIONAL_DESPAIR: "EMOTIONAL_DRIVE",
  EMOTIONAL_SUPPORT: "EMOTIONAL_DRIVE", EMOTIONAL_APOLOGY: "EMOTIONAL_DRIVE",
  EMOTIONAL_CONFESSION: "EMOTIONAL_DRIVE", EMOTIONAL_ENDURANCE: "EMOTIONAL_DRIVE",
  PSYCHOLOGICAL_IMPACT: "EMOTIONAL_DRIVE", PROTECTIVE_INSTINCT: "EMOTIONAL_DRIVE",
  CRUELTY_PLEASURE: "EMOTIONAL_DRIVE", NOSTALGIA_INDUCEMENT: "EMOTIONAL_DRIVE",
  ENRAGES: "EMOTIONAL_DRIVE",
  PSYCHOLOGICAL_PRESSURE: "EMOTIONAL_DRIVE",
  PSYCHOLOGICAL_REINFORCEMENT: "EMOTIONAL_DRIVE", EMOTIONAL_DISTANCE: "EMOTIONAL_DRIVE",
  // Social bond
  ALLY_DEPENDENCE: "SOCIAL_BOND", ALLY_SUPPORT: "SOCIAL_BOND",
  FAMILY_INFLUENCE: "SOCIAL_BOND", FAMILY_BACKGROUND_REACTION: "SOCIAL_BOND",
  INHERITED_OBLIGATION: "SOCIAL_BOND", MENTORSHIP_SUPPORT: "SOCIAL_BOND",
  MOTIVATES: "SOCIAL_BOND", PERSUASION_ATTEMPT: "SOCIAL_BOND",
  INTERPERSONAL_CARE: "SOCIAL_BOND", MORAL_GUIDANCE: "SOCIAL_BOND",
  INTERPERSONAL_BOUNDARY: "SOCIAL_BOND",
  // Narrative escalation
  CAUSES_REVERSAL: "NARRATIVE_ESCALATION", ACTION_ESCALATION: "NARRATIVE_ESCALATION",
  CONSCIENCE_CONFLICT: "NARRATIVE_ESCALATION", IDENTITY_CONFLICT: "NARRATIVE_ESCALATION",
  CONFLICT_OF_INTEREST: "NARRATIVE_ESCALATION", PHYSICAL_CONFRONTATION: "NARRATIVE_ESCALATION",
  ESCALATES: "NARRATIVE_ESCALATION", COMPLICATES_FURTHER: "NARRATIVE_ESCALATION",
  CHALLENGES: "NARRATIVE_ESCALATION", MORAL_CHALLENGE: "NARRATIVE_ESCALATION",
  MISSED_OPPORTUNITY: "NARRATIVE_ESCALATION",
  EXPECTATION_DISAPPOINTMENT: "NARRATIVE_ESCALATION",
  PERSONAL_TRANSFORMATION: "NARRATIVE_ESCALATION",
  PERCEPTION_SHIFT: "NARRATIVE_ESCALATION",
  INTERPERSONAL_CONFLICT: "NARRATIVE_ESCALATION", ESCALATES_CONFLICT: "NARRATIVE_ESCALATION",
  SCENE_REVERSAL: "NARRATIVE_ESCALATION", MORAL_CORRUPTION_INFLUENCE: "NARRATIVE_ESCALATION",
  LEADS_TO_CRISIS: "NARRATIVE_ESCALATION", EXPECTED_RESULT_SHIFT: "NARRATIVE_ESCALATION",
  // Resolution
  RESOLVES: "NARRATIVE_RESOLUTION", CONCLUDES: "NARRATIVE_RESOLUTION",
  REDEEMS: "NARRATIVE_RESOLUTION", PERSONAL_JOURNEY: "NARRATIVE_RESOLUTION",
  MENTAL_RELIEF: "NARRATIVE_RESOLUTION",
  // Revelation / epistemic
  REVEALS: "REVELATION_EPISTEMIC", EXPOSES: "REVELATION_EPISTEMIC",
  CONCEALS: "REVELATION_EPISTEMIC", FORESHADOWS: "REVELATION_EPISTEMIC",
  PAST_CONNECTION: "REVELATION_EPISTEMIC", LOVE_INSIGHT: "REVELATION_EPISTEMIC",
  HISTORICAL_COMPARISON: "REVELATION_EPISTEMIC",
  REVEALS_INFORMATION: "REVELATION_EPISTEMIC", BACKSTORY_PRESSURE: "REVELATION_EPISTEMIC",
  MORAL_REVELATION_TRIGGER: "REVELATION_EPISTEMIC", MORAL_JUDGMENT: "REVELATION_EPISTEMIC",
  // Mediation / transfer
  INFORMS: "MEDIATION_TRANSFER", MEDIATES: "MEDIATION_TRANSFER",
  TRANSFERS: "MEDIATION_TRANSFER", DELEGATES: "MEDIATION_TRANSFER",
  FINANCIAL_NEED: "MEDIATION_TRANSFER", CULTURAL_EDUCATION: "MEDIATION_TRANSFER",
  DECISION_MAKING: "MEDIATION_TRANSFER",
  // Thematic
  CONTRASTS: "THEMATIC_CONTRAST", MIRRORS: "THEMATIC_CONTRAST",
  EXPLAINS: "THEMATIC_EXPLANATION", SUPPORTS: "THEMATIC_EXPLANATION",
  NARRATIVE_COMPOSITE: "THEMATIC_EXPLANATION",
};

// Register Cytoscape extensions if their globals loaded.
if (typeof cytoscape !== "undefined") {
  if (typeof cytoscapeFcose !== "undefined") cytoscape.use(cytoscapeFcose);
  if (typeof cytoscapeCola !== "undefined") cytoscape.use(cytoscapeCola);
}

const state = {
  manifest: null,
  novelKey: null,
  events: [],
  eventById: new Map(),
  causalEdges: [],
  liveLayout: null,
  thematicEdges: [],
  agentToEvents: new Map(),
  themeToEvents: { POWER: new Set(), WEALTH: new Set(), KINSHIP: new Set(), JUSTICE: new Set(), KNOWLEDGE: new Set() },
  chapterMin: 1,
  chapterMax: 1,
  cy: null,
  selectedEventId: null,
};

const ui = {
  novelSelect: document.getElementById("novel-select"),
  search: document.getElementById("search"),
  statEvents: document.getElementById("stat-events"),
  statEdges: document.getElementById("stat-edges"),
  statShown: document.getElementById("stat-shown"),
  statStatus: document.getElementById("stat-status"),
  themeFilters: document.getElementById("theme-filters"),
  themeConfidence: document.getElementById("theme-confidence"),
  themeConfidenceVal: document.getElementById("theme-confidence-val"),
  viewMode: document.getElementById("view-mode"),
  subplotSection: document.getElementById("subplot-section"),
  subplotTheme: document.getElementById("subplot-theme"),
  subplotInfo: document.getElementById("subplot-info"),
  chapterMin: document.getElementById("chapter-min"),
  chapterMax: document.getElementById("chapter-max"),
  agentSelect: document.getElementById("agent-select"),
  showCausal: document.getElementById("show-causal"),
  showThematic: document.getElementById("show-thematic"),
  edgeConfidence: document.getElementById("edge-confidence"),
  edgeConfidenceVal: document.getElementById("edge-confidence-val"),
  supertypeLegend: document.getElementById("supertype-legend"),
  layoutSelect: document.getElementById("layout-select"),
  reLayout: document.getElementById("re-layout"),
  fitView: document.getElementById("fit-view"),
  maxEvents: document.getElementById("max-events"),
  graph: document.getElementById("graph"),
  graphEmpty: document.getElementById("graph-empty"),
  detailEmpty: document.getElementById("detail-empty"),
  detailContent: document.getElementById("detail-content"),
};

// ---------- Loading ----------

async function init() {
  ui.statStatus.textContent = "loading manifest…";
  let manifest;
  try {
    const res = await fetch("data/manifest.json", { cache: "no-cache" });
    manifest = await res.json();
  } catch (e) {
    ui.statStatus.textContent = "no manifest found";
    return;
  }
  state.manifest = manifest;
  for (const novel of manifest.novels) {
    const opt = document.createElement("option");
    opt.value = novel.key;
    opt.textContent = novel.label;
    ui.novelSelect.appendChild(opt);
  }
  ui.novelSelect.value = manifest.default || manifest.novels[0].key;
  ui.novelSelect.addEventListener("change", () => loadNovel(ui.novelSelect.value));
  await loadNovel(ui.novelSelect.value);
}

async function loadNovel(key) {
  ui.statStatus.textContent = `loading ${key}…`;
  state.novelKey = key;
  const novel = state.manifest.novels.find(n => n.key === key);
  const res = await fetch(`data/${novel.file}`, { cache: "no-cache" });
  const data = await res.json();
  ingest(data);
  ui.statStatus.textContent = `loaded ${key}`;
  applyFilters();
}

function ingest(jsonld) {
  const graph = jsonld["@graph"] || [];
  state.events = [];
  state.eventById.clear();
  state.causalEdges = [];
  state.thematicEdges = [];
  state.agentToEvents = new Map();
  for (const t of THEMES) state.themeToEvents[t] = new Set();

  for (const item of graph) {
    if (item.type === "Event") {
      const ev = {
        id: item["@id"],
        description: item.raw_description || "",
        actionType: item.action_type || "",
        chapter: item.chapter || 0,
        sequence: item.sequence || 0,
        location: item.location_context || "",
        time: item.time_context || "",
        actors: item.actors || [],
        patients: item.patients || [],
        whyFactors: item.why_factors || [],
        sourceQuote: item.source_quote || "",
        confidence: item.confidence ?? 1.0,
        sceneId: item.scene_id || null,
        themes: item.theme_annotations || {},
      };
      state.events.push(ev);
      state.eventById.set(ev.id, ev);
      for (const a of ev.actors) addToBucket(state.agentToEvents, a, ev.id);
      for (const p of ev.patients) addToBucket(state.agentToEvents, p, ev.id);
      for (const t of THEMES) {
        const td = ev.themes[t];
        if (td && (td.involvement === "direct" || td.involvement === "indirect")) {
          state.themeToEvents[t].add(ev.id);
        }
      }
    } else if (item.type === "CausalEdge") {
      const rt = (item.relationType || "").toUpperCase();
      const supertype = item.edge_supertype || RELTYPE_TO_SUPERTYPE[rt] || null;
      state.causalEdges.push({
        id: item["@id"],
        from: item.from, to: item.to,
        relationType: item.relationType || "",
        mechanism: item.mechanism || "",
        weight: item.weight ?? 1.0,
        confidence: item.confidence ?? 1.0,
        edgeSupertype: supertype,
      });
    } else if (item.type === "ThematicEdge") {
      state.thematicEdges.push({
        id: item["@id"],
        from: item.from, to: item.to,
        theme: item.theme || "",
        sourceRole: item.source_role || "",
        targetRole: item.target_role || "",
        confidence: item.confidence ?? null,
        via: item.via || "",
      });
    }
  }

  const chapters = state.events.map(e => e.chapter).filter(c => c > 0);
  state.chapterMin = chapters.length ? Math.min(...chapters) : 1;
  state.chapterMax = chapters.length ? Math.max(...chapters) : 1;

  buildSidebar();
  ui.statEvents.textContent = `events: ${state.events.length.toLocaleString()}`;
  ui.statEdges.textContent = `edges: ${(state.causalEdges.length + state.thematicEdges.length).toLocaleString()}`;
}

function addToBucket(map, key, value) {
  if (!key) return;
  if (!map.has(key)) map.set(key, new Set());
  map.get(key).add(value);
}

// ---------- Sidebar wiring ----------

function buildSidebar() {
  // Causal-edge supertype legend, restricted to supertypes actually present in this dataset.
  const presentSupertypes = new Map();
  for (const e of state.causalEdges) {
    const st = e.edgeSupertype || "OTHER";
    presentSupertypes.set(st, (presentSupertypes.get(st) || 0) + 1);
  }
  const legendEntries = [...presentSupertypes.entries()].sort((a, b) => b[1] - a[1]);
  ui.supertypeLegend.innerHTML = "";
  for (const [st, count] of legendEntries) {
    const color = SUPERTYPE_COLOR[st] || SUPERTYPE_DEFAULT;
    const row = document.createElement("div");
    row.className = "legend-row";
    row.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>
                     <span>${escapeHtml(st)}</span>
                     <span class="hint">(${count.toLocaleString()})</span>`;
    ui.supertypeLegend.appendChild(row);
  }
  if (state.thematicEdges.length) {
    const row = document.createElement("div");
    row.className = "legend-row dashed";
    row.style.marginTop = "6px";
    row.innerHTML = `<span class="legend-swatch" style="background:#888"></span>
                     <span>THEMATIC (dashed)</span>
                     <span class="hint">(${state.thematicEdges.length.toLocaleString()})</span>`;
    ui.supertypeLegend.appendChild(row);
  }

  ui.themeFilters.innerHTML = "";
  for (const t of THEMES) {
    const lbl = document.createElement("label");
    lbl.innerHTML = `<input type="checkbox" data-theme="${t}" checked> <span class="theme-${t.toLowerCase()}">${t}</span> <span class="hint">(${state.themeToEvents[t].size})</span>`;
    lbl.querySelector("input").addEventListener("change", applyFilters);
    ui.themeFilters.appendChild(lbl);
  }

  ui.subplotTheme.innerHTML = "";
  for (const t of THEMES) {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = `${t} (${state.themeToEvents[t].size})`;
    ui.subplotTheme.appendChild(opt);
  }

  ui.chapterMin.value = state.chapterMin;
  ui.chapterMin.min = state.chapterMin;
  ui.chapterMin.max = state.chapterMax;
  // Show 3 chapters by default — wide enough to see edge-type variety,
  // narrow enough that the layout settles fast on first paint.
  ui.chapterMax.value = Math.min(state.chapterMin + 2, state.chapterMax);
  ui.chapterMax.min = state.chapterMin;
  ui.chapterMax.max = state.chapterMax;

  // Agents sorted by event count
  const sortedAgents = [...state.agentToEvents.entries()]
    .sort((a, b) => b[1].size - a[1].size);
  ui.agentSelect.innerHTML = '<option value="">— any —</option>';
  for (const [agent, evs] of sortedAgents) {
    const opt = document.createElement("option");
    opt.value = agent;
    opt.textContent = `${agent} (${evs.size})`;
    ui.agentSelect.appendChild(opt);
  }
}

ui.themeConfidence.addEventListener("input", () => {
  ui.themeConfidenceVal.textContent = parseFloat(ui.themeConfidence.value).toFixed(2);
  applyFilters();
});
ui.viewMode.addEventListener("change", () => {
  ui.subplotSection.hidden = ui.viewMode.value !== "subplot";
  applyFilters();
});
ui.subplotTheme.addEventListener("change", applyFilters);
ui.chapterMin.addEventListener("change", applyFilters);
ui.chapterMax.addEventListener("change", applyFilters);
ui.agentSelect.addEventListener("change", applyFilters);
ui.showCausal.addEventListener("change", applyFilters);
ui.showThematic.addEventListener("change", applyFilters);
ui.edgeConfidence.addEventListener("input", () => {
  ui.edgeConfidenceVal.textContent = parseFloat(ui.edgeConfidence.value).toFixed(2);
  applyFilters();
});
ui.maxEvents.addEventListener("change", applyFilters);
ui.search.addEventListener("input", debounce(applyFilters, 250));
ui.layoutSelect.addEventListener("change", () => runLayout());
ui.reLayout.addEventListener("click", () => runLayout());
ui.fitView.addEventListener("click", () => state.cy && state.cy.fit(null, 30));

function debounce(fn, ms) {
  let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---------- Filter / render ----------

function applyFilters() {
  const themesEnabled = new Set(
    [...ui.themeFilters.querySelectorAll("input[type=checkbox]")]
      .filter(cb => cb.checked).map(cb => cb.dataset.theme)
  );
  const minConf = parseFloat(ui.themeConfidence.value);
  const chMin = parseInt(ui.chapterMin.value, 10);
  const chMax = parseInt(ui.chapterMax.value, 10);
  const agent = ui.agentSelect.value;
  const search = ui.search.value.trim().toLowerCase();
  const view = ui.viewMode.value;
  const subplotTheme = ui.subplotTheme.value;
  const showCausal = ui.showCausal.checked;
  const showThematic = ui.showThematic.checked;
  const minEdgeConf = parseFloat(ui.edgeConfidence.value);
  const maxEvents = parseInt(ui.maxEvents.value, 10);

  // Event filter
  let visible = new Set();
  for (const ev of state.events) {
    if (ev.chapter < chMin || ev.chapter > chMax) continue;
    if (agent) {
      if (!ev.actors.includes(agent) && !ev.patients.includes(agent)) continue;
    }
    if (search && !ev.description.toLowerCase().includes(search) && !ev.sourceQuote.toLowerCase().includes(search)) continue;

    // Theme involvement is only treated as a *filter* in subplot view.
    // In causal / agent views, the theme checkboxes and confidence slider
    // only influence node coloring (see dominantThemeColor below). This
    // matches reader intuition: a routine walk-across-the-room event with
    // all-"none" themes is still an event worth showing in the causal graph.
    if (view === "subplot") {
      const td = ev.themes[subplotTheme];
      if (!td) continue;
      if (td.involvement !== "direct" && td.involvement !== "indirect") continue;
      if ((td.confidence ?? 0) < minConf) continue;
    }

    visible.add(ev.id);
    if (visible.size >= maxEvents) break;
  }

  // Edge filter — both endpoints must be visible, and edge confidence must clear the slider.
  const causalShown = !showCausal ? [] : state.causalEdges.filter(e =>
    visible.has(e.from) && visible.has(e.to) && (e.confidence ?? 0) >= minEdgeConf
  );
  const thematicFiltered = !showThematic ? [] : state.thematicEdges.filter(e => {
    if (!visible.has(e.from) || !visible.has(e.to)) return false;
    if (view === "subplot" && e.theme !== subplotTheme) return false;
    if ((e.confidence ?? 0) < minEdgeConf) return false;
    return true;
  });

  // In subplot view, drop causal edges (we want clean subplot rendering)
  const finalCausal = view === "subplot" ? [] : causalShown;
  const finalThematic = view === "subplot" || showThematic ? thematicFiltered : [];

  ui.statShown.textContent = `shown: ${visible.size.toLocaleString()} / ${state.events.length.toLocaleString()}`;
  if (view === "subplot") {
    const totalForTheme = state.themeToEvents[subplotTheme]?.size || 0;
    ui.subplotInfo.textContent = `${totalForTheme} events have ${subplotTheme} involvement (any chapter, any confidence)`;
  }
  render(visible, finalCausal, finalThematic, themesEnabled, minConf);
}

function render(visibleSet, causalEdges, thematicEdges, themesEnabled, minConf) {
  if (visibleSet.size === 0) {
    ui.graphEmpty.hidden = false;
    if (state.cy) state.cy.elements().remove();
    return;
  }
  ui.graphEmpty.hidden = true;

  const nodes = [];
  for (const id of visibleSet) {
    const ev = state.eventById.get(id);
    nodes.push({
      data: {
        id: ev.id,
        label: truncate(ev.description, 60),
        chapter: ev.chapter,
        themeColor: dominantThemeColor(ev, themesEnabled, minConf),
      },
    });
  }
  const edges = [];
  for (const e of causalEdges) {
    const conf = clamp01(e.confidence ?? 0.5);
    edges.push({
      data: {
        id: e.id, source: e.from, target: e.to,
        relType: e.relationType, kind: "causal",
        confidence: conf,
        supertype: e.edgeSupertype || "OTHER",
        edgeColor: SUPERTYPE_COLOR[e.edgeSupertype] || SUPERTYPE_DEFAULT,
        // Cose layout reads `weight` as a spring constant — high confidence
        // pulls endpoints closer, so well-supported causes sit near their effects.
        weight: 0.5 + conf * 1.5,
      },
    });
  }
  for (const e of thematicEdges) {
    const conf = clamp01(e.confidence ?? 0.5);
    edges.push({
      data: {
        id: e.id, source: e.from, target: e.to,
        relType: e.theme, kind: "thematic",
        confidence: conf,
        edgeColor: THEME_COLOR[e.theme] || "#888",
        weight: 0.3 + conf * 1.2,
      },
    });
  }

  const elements = [...nodes, ...edges];
  if (!state.cy) {
    state.cy = cytoscape({
      container: ui.graph,
      elements,
      wheelSensitivity: 0.3,
      style: cyStyle(),
    });
    state.cy.on("tap", "node", evt => showDetail(evt.target.id()));
    state.cy.on("tap", "edge", evt => showEdgeDetail(evt.target.data()));
  } else {
    state.cy.elements().remove();
    state.cy.add(elements);
  }
  runLayout();
}

function cyStyle() {
  return [
    {
      selector: "node",
      style: {
        "background-color": "data(themeColor)",
        "label": "data(label)",
        "color": "#fff",
        "font-size": 8,
        "text-valign": "bottom",
        "text-halign": "center",
        "text-margin-y": 3,
        "width": 16, "height": 16,
        "border-color": "#0a0c12", "border-width": 1,
      },
    },
    {
      selector: "node:selected",
      style: { "border-color": "#fff", "border-width": 3, "width": 22, "height": 22 },
    },
    {
      // Causal edges: width and opacity scale with confidence;
      // color encodes edge_supertype (production / constraint / revelation / etc).
      // mapData range tightened to the actual gpt-5 data distribution
      // (causal confidence almost always falls in [0.4, 0.9]) so the visual
      // delta between low- and high-confidence edges is visible.
      selector: "edge[kind = 'causal']",
      style: {
        "width": "mapData(confidence, 0.4, 0.9, 0.6, 5)",
        "opacity": "mapData(confidence, 0.4, 0.9, 0.35, 1)",
        "line-color": "data(edgeColor)",
        "target-arrow-color": "data(edgeColor)",
        "target-arrow-shape": "triangle",
        "arrow-scale": 0.8,
        "curve-style": "bezier",
      },
    },
    {
      // Thematic edges: theme color, dashed, also width/opacity by confidence.
      selector: "edge[kind = 'thematic']",
      style: {
        "width": "mapData(confidence, 0.3, 0.9, 0.8, 5.5)",
        "opacity": "mapData(confidence, 0.3, 0.9, 0.4, 1)",
        "line-color": "data(edgeColor)",
        "target-arrow-color": "data(edgeColor)",
        "target-arrow-shape": "triangle",
        "arrow-scale": 0.8,
        "line-style": "dashed",
        "curve-style": "bezier",
      },
    },
    {
      selector: "edge:selected",
      style: { "width": 5, "opacity": 1, "z-index": 999 },
    },
  ];
}

function clamp01(x) { return Math.max(0, Math.min(1, +x || 0)); }

function runLayout() {
  if (!state.cy) return;

  // Stop any continuous physics simulation from a prior layout.
  if (state.liveLayout) {
    try { state.liveLayout.stop(); } catch (e) {}
    state.liveLayout = null;
  }

  const name = ui.layoutSelect.value;
  const opts = {
    // fcose: well-spaced force-directed; default. Edge `weight` (set from
    // confidence) feeds into the spring constants, so high-confidence edges
    // are shorter and low-confidence edges are longer — visual "distance".
    fcose: {
      name: "fcose",
      animate: true,
      animationDuration: 600,
      randomize: true,
      quality: "default",
      nodeRepulsion: 8000,
      idealEdgeLength: edge => 80 + (1 - (edge.data("confidence") || 0.5)) * 220,
      edgeElasticity: 0.45,
      gravity: 0.18,
      gravityRangeCompound: 1.5,
      nodeSeparation: 80,
      packComponents: true,
      padding: 40,
    },
    // cola with infinite:true keeps the physics running. Drag a node and
    // its neighbors react in real time. Costlier above ~1500 nodes; the
    // sidebar's max-events cap keeps it usable.
    "cola-live": {
      name: "cola",
      animate: true,
      infinite: true,
      fit: false,
      randomize: false,
      avoidOverlap: true,
      handleDisconnected: true,
      nodeSpacing: 18,
      edgeLength: edge => 70 + (1 - (edge.data("confidence") || 0.5)) * 200,
      edgeSymDiffLength: 30,
    },
    // Vanilla cose, retuned for much more spread than the Cytoscape default.
    cose: {
      name: "cose",
      animate: "end",
      animationDuration: 600,
      idealEdgeLength: 220,
      nodeRepulsion: 800000,
      nodeOverlap: 24,
      gravity: 30,
      numIter: 1500,
      padding: 40,
    },
    dagre: { name: "dagre", rankDir: "LR", nodeSep: 30, rankSep: 90, animate: true, animationDuration: 500, padding: 30 },
    grid: { name: "grid", padding: 30, animate: true, animationDuration: 400, avoidOverlap: true },
    concentric: {
      name: "concentric",
      concentric: n => -n.data("chapter"),
      levelWidth: () => 1,
      minNodeSpacing: 18,
      animate: true, animationDuration: 500, padding: 30,
    },
  };
  const cfg = opts[name] || opts.fcose;
  const layout = state.cy.layout(cfg);
  layout.run();
  if (cfg.infinite) state.liveLayout = layout;
}

function dominantThemeColor(ev, themesEnabled, minConf) {
  // Pick the theme this event is most strongly involved with — but only
  // among the themes the user currently has enabled, and above the
  // confidence threshold. Events that don't qualify get a neutral color
  // and stay visible (filtering happens elsewhere).
  let bestT = null, bestC = 0;
  for (const t of THEMES) {
    if (themesEnabled && !themesEnabled.has(t)) continue;
    const td = ev.themes[t];
    if (!td) continue;
    if (td.involvement !== "direct" && td.involvement !== "indirect") continue;
    const c = td.confidence ?? 0;
    if (c < (minConf ?? 0)) continue;
    if (c > bestC) { bestC = c; bestT = t; }
  }
  return bestT ? THEME_COLOR[bestT] : ACTION_COLOR_DEFAULT;
}

function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

// ---------- Detail panel ----------

function showDetail(eventId) {
  const ev = state.eventById.get(eventId);
  if (!ev) return;
  state.selectedEventId = eventId;
  ui.detailEmpty.hidden = true;
  ui.detailContent.hidden = false;

  const themeRows = THEMES.map(t => {
    const td = ev.themes[t] || { involvement: "none", role: null, confidence: 0, evidence: "" };
    const conf = (td.confidence ?? 0);
    const pct = Math.round(conf * 100);
    return `
      <tr>
        <td class="theme-name theme-${t.toLowerCase()}">${t}</td>
        <td>
          <span class="theme-bar"><span class="fill-${t.toLowerCase()}" style="width:${pct}%"></span></span>
          <span class="invol-${td.involvement}">${td.involvement}${td.role ? " · " + td.role : ""}</span>
        </td>
        <td>${conf.toFixed(2)}</td>
      </tr>
      ${td.evidence ? `<tr><td colspan="3" class="hint" style="padding-left:10px;font-style:italic">${escapeHtml(td.evidence)}</td></tr>` : ""}
    `;
  }).join("");

  const causalIn = state.causalEdges.filter(e => e.to === eventId);
  const causalOut = state.causalEdges.filter(e => e.from === eventId);

  ui.detailContent.innerHTML = `
    <h2>${escapeHtml(truncate(ev.description, 90))}</h2>
    <div class="meta-row">Ch ${ev.chapter} · seq ${ev.sequence} · ${escapeHtml(ev.actionType)} · conf ${ev.confidence.toFixed(2)}</div>
    ${ev.sourceQuote ? `<div class="quote" style="margin-top:8px">${escapeHtml(truncate(ev.sourceQuote, 350))}</div>` : ""}

    <div class="detail-section">
      <div class="panel-title">Themes</div>
      <table class="themes">${themeRows}</table>
    </div>

    ${ev.actors.length ? `
    <div class="detail-section">
      <div class="panel-title">Actors</div>
      <div class="chips">${ev.actors.map(a => `<span class="chip" data-agent="${escapeHtml(a)}">${escapeHtml(a)}</span>`).join("")}</div>
    </div>` : ""}

    ${ev.patients.length ? `
    <div class="detail-section">
      <div class="panel-title">Patients</div>
      <div class="chips">${ev.patients.map(a => `<span class="chip" data-agent="${escapeHtml(a)}">${escapeHtml(a)}</span>`).join("")}</div>
    </div>` : ""}

    ${ev.whyFactors.length ? `
    <div class="detail-section">
      <div class="panel-title">Why factors</div>
      <div class="chips">${ev.whyFactors.map(w => `<span class="chip">${escapeHtml(w)}</span>`).join("")}</div>
    </div>` : ""}

    <div class="detail-section">
      <div class="panel-title">Caused by (${causalIn.length})</div>
      ${causalIn.slice(0, 8).map(e => {
        const src = state.eventById.get(e.from);
        return `<div class="neighbor" data-jump="${e.from}"><span class="rel">${escapeHtml(e.relationType)}</span> ${escapeHtml(truncate(src ? src.description : e.from, 70))}</div>`;
      }).join("") || `<div class="hint">none in current data</div>`}
    </div>

    <div class="detail-section">
      <div class="panel-title">Causes (${causalOut.length})</div>
      ${causalOut.slice(0, 8).map(e => {
        const dst = state.eventById.get(e.to);
        return `<div class="neighbor" data-jump="${e.to}"><span class="rel">${escapeHtml(e.relationType)}</span> ${escapeHtml(truncate(dst ? dst.description : e.to, 70))}</div>`;
      }).join("") || `<div class="hint">none in current data</div>`}
    </div>
  `;

  ui.detailContent.querySelectorAll("[data-jump]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.jump;
      if (state.cy && state.cy.getElementById(id).length) {
        state.cy.center(state.cy.getElementById(id));
      }
      showDetail(id);
    });
  });
  ui.detailContent.querySelectorAll("[data-agent]").forEach(el => {
    el.addEventListener("click", () => {
      ui.agentSelect.value = el.dataset.agent;
      applyFilters();
    });
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function showEdgeDetail(d) {
  const src = state.eventById.get(d.source);
  const tgt = state.eventById.get(d.target);
  const conf = (d.confidence ?? 0).toFixed(2);
  const colorChip = `<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:${d.edgeColor};vertical-align:middle;margin-right:6px"></span>`;
  const kindLabel = d.kind === "causal"
    ? `${colorChip}${escapeHtml(d.relType)}${d.supertype && d.supertype !== "OTHER" ? ` <span class="hint">(${escapeHtml(d.supertype)})</span>` : ""}`
    : `${colorChip}THEMATIC · ${escapeHtml(d.relType)}`;

  ui.detailEmpty.hidden = true;
  ui.detailContent.hidden = false;
  ui.detailContent.innerHTML = `
    <h2>Edge</h2>
    <div class="meta-row">${kindLabel}</div>
    <div class="meta-row">confidence: <strong>${conf}</strong></div>
    <div class="detail-section">
      <div class="panel-title">From</div>
      <div class="neighbor" data-jump="${d.source}">${escapeHtml(truncate(src ? src.description : d.source, 90))}</div>
      <div class="panel-title" style="margin-top:8px">To</div>
      <div class="neighbor" data-jump="${d.target}">${escapeHtml(truncate(tgt ? tgt.description : d.target, 90))}</div>
    </div>
  `;
  ui.detailContent.querySelectorAll("[data-jump]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.jump;
      if (state.cy && state.cy.getElementById(id).length) state.cy.center(state.cy.getElementById(id));
      showDetail(id);
    });
  });
}

init();
