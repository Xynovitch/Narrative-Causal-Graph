/* CEKG webpage. Loads a pipeline JSON-LD file and renders an interactive view. */

const THEMES = ["POWER", "WEALTH", "KINSHIP", "JUSTICE", "KNOWLEDGE"];
const THEME_COLOR = {
  POWER: "#e15759", WEALTH: "#f1ce63", KINSHIP: "#59a14f",
  JUSTICE: "#8cd17d", KNOWLEDGE: "#4e79a7",
};
const ACTION_COLOR_DEFAULT = "#7a8aa8";

const SUPERTYPE_COLOR = {
  CAUSAL_PRODUCTION:    "#4eae5b",
  CAUSAL_CONSTRAINT:    "#d24747",
  EMOTIONAL_DRIVE:      "#e98abf",
  SOCIAL_BOND:          "#e8973c",
  NARRATIVE_ESCALATION: "#e15759",
  NARRATIVE_RESOLUTION: "#76b7e0",
  REVELATION_EPISTEMIC: "#9b6bd9",
  MEDIATION_TRANSFER:   "#f1ce63",
  THEMATIC_CONTRAST:    "#7ec6c2",
  THEMATIC_EXPLANATION: "#a3b87f",
};
const SUPERTYPE_DEFAULT = "#5a607a";
const CHRONO_COLOR = "#6b7390";

const RELTYPE_TO_SUPERTYPE = {
  CAUSES: "CAUSAL_PRODUCTION", DIRECT_CAUSE: "CAUSAL_PRODUCTION",
  ENABLES: "CAUSAL_PRODUCTION", FACILITATES: "CAUSAL_PRODUCTION",
  TRIGGERS: "CAUSAL_PRODUCTION", INCITING_CAUSE: "CAUSAL_PRODUCTION",
  EVENT_ENABLES_NEXT: "CAUSAL_PRODUCTION", EVENT_REINFORCEMENT: "CAUSAL_PRODUCTION",
  DESIRE_ALIGNMENT: "CAUSAL_PRODUCTION", NECESSITATES: "CAUSAL_PRODUCTION",
  FULFILLS: "CAUSAL_PRODUCTION", PRECEDES: "CAUSAL_PRODUCTION",
  SCENE_CAUSATION: "CAUSAL_PRODUCTION", SCENE_CHAINING: "CAUSAL_PRODUCTION",
  PLOT_PROPULSION: "CAUSAL_PRODUCTION", STRUCTURAL_DEPENDENCE: "CAUSAL_PRODUCTION",
  CONSEQUENCE_CHAINING: "CAUSAL_PRODUCTION", REINFORCES_GOAL: "CAUSAL_PRODUCTION",
  PREVENTS: "CAUSAL_CONSTRAINT", BLOCKS: "CAUSAL_CONSTRAINT", INHIBITS: "CAUSAL_CONSTRAINT",
  COMPLICATES: "CAUSAL_CONSTRAINT", OPPOSES: "CAUSAL_CONSTRAINT",
  DESIRE_OBSTRUCTION: "CAUSAL_CONSTRAINT", DESIRE_COMPETITION: "CAUSAL_CONSTRAINT",
  PHYSICAL_BLOCKAGE: "CAUSAL_CONSTRAINT", INTERRUPTION_OBSTACLE: "CAUSAL_CONSTRAINT",
  MISSION_FAILURE: "CAUSAL_CONSTRAINT", MISSION_ABANDONMENT: "CAUSAL_CONSTRAINT",
  OPPOSITION_PRESSURE: "CAUSAL_CONSTRAINT", PREVENTS_OUTCOME: "CAUSAL_CONSTRAINT",
  RELATIONAL_FRAGMENTATION: "CAUSAL_CONSTRAINT",
  COMPASSION_TRIGGER: "EMOTIONAL_DRIVE", EMOTIONAL_MANIPULATION: "EMOTIONAL_DRIVE",
  EMOTIONAL_DEPENDENCE: "EMOTIONAL_DRIVE", EMOTIONAL_TRIGGER: "EMOTIONAL_DRIVE",
  EMOTIONAL_CONTAGION: "EMOTIONAL_DRIVE", EMOTIONAL_DESPAIR: "EMOTIONAL_DRIVE",
  EMOTIONAL_SUPPORT: "EMOTIONAL_DRIVE", EMOTIONAL_APOLOGY: "EMOTIONAL_DRIVE",
  EMOTIONAL_CONFESSION: "EMOTIONAL_DRIVE", EMOTIONAL_ENDURANCE: "EMOTIONAL_DRIVE",
  PSYCHOLOGICAL_IMPACT: "EMOTIONAL_DRIVE", PROTECTIVE_INSTINCT: "EMOTIONAL_DRIVE",
  CRUELTY_PLEASURE: "EMOTIONAL_DRIVE", NOSTALGIA_INDUCEMENT: "EMOTIONAL_DRIVE",
  ENRAGES: "EMOTIONAL_DRIVE", PSYCHOLOGICAL_PRESSURE: "EMOTIONAL_DRIVE",
  PSYCHOLOGICAL_REINFORCEMENT: "EMOTIONAL_DRIVE", EMOTIONAL_DISTANCE: "EMOTIONAL_DRIVE",
  ALLY_DEPENDENCE: "SOCIAL_BOND", ALLY_SUPPORT: "SOCIAL_BOND",
  FAMILY_INFLUENCE: "SOCIAL_BOND", FAMILY_BACKGROUND_REACTION: "SOCIAL_BOND",
  INHERITED_OBLIGATION: "SOCIAL_BOND", MENTORSHIP_SUPPORT: "SOCIAL_BOND",
  MOTIVATES: "SOCIAL_BOND", PERSUASION_ATTEMPT: "SOCIAL_BOND",
  INTERPERSONAL_CARE: "SOCIAL_BOND", MORAL_GUIDANCE: "SOCIAL_BOND",
  INTERPERSONAL_BOUNDARY: "SOCIAL_BOND",
  CAUSES_REVERSAL: "NARRATIVE_ESCALATION", ACTION_ESCALATION: "NARRATIVE_ESCALATION",
  CONSCIENCE_CONFLICT: "NARRATIVE_ESCALATION", IDENTITY_CONFLICT: "NARRATIVE_ESCALATION",
  CONFLICT_OF_INTEREST: "NARRATIVE_ESCALATION", PHYSICAL_CONFRONTATION: "NARRATIVE_ESCALATION",
  ESCALATES: "NARRATIVE_ESCALATION", COMPLICATES_FURTHER: "NARRATIVE_ESCALATION",
  CHALLENGES: "NARRATIVE_ESCALATION", MORAL_CHALLENGE: "NARRATIVE_ESCALATION",
  MISSED_OPPORTUNITY: "NARRATIVE_ESCALATION", EXPECTATION_DISAPPOINTMENT: "NARRATIVE_ESCALATION",
  PERSONAL_TRANSFORMATION: "NARRATIVE_ESCALATION", PERCEPTION_SHIFT: "NARRATIVE_ESCALATION",
  INTERPERSONAL_CONFLICT: "NARRATIVE_ESCALATION", ESCALATES_CONFLICT: "NARRATIVE_ESCALATION",
  SCENE_REVERSAL: "NARRATIVE_ESCALATION", MORAL_CORRUPTION_INFLUENCE: "NARRATIVE_ESCALATION",
  LEADS_TO_CRISIS: "NARRATIVE_ESCALATION", EXPECTED_RESULT_SHIFT: "NARRATIVE_ESCALATION",
  RESOLVES: "NARRATIVE_RESOLUTION", CONCLUDES: "NARRATIVE_RESOLUTION",
  REDEEMS: "NARRATIVE_RESOLUTION", PERSONAL_JOURNEY: "NARRATIVE_RESOLUTION",
  MENTAL_RELIEF: "NARRATIVE_RESOLUTION",
  REVEALS: "REVELATION_EPISTEMIC", EXPOSES: "REVELATION_EPISTEMIC",
  CONCEALS: "REVELATION_EPISTEMIC", FORESHADOWS: "REVELATION_EPISTEMIC",
  PAST_CONNECTION: "REVELATION_EPISTEMIC", LOVE_INSIGHT: "REVELATION_EPISTEMIC",
  HISTORICAL_COMPARISON: "REVELATION_EPISTEMIC", REVEALS_INFORMATION: "REVELATION_EPISTEMIC",
  BACKSTORY_PRESSURE: "REVELATION_EPISTEMIC", MORAL_REVELATION_TRIGGER: "REVELATION_EPISTEMIC",
  MORAL_JUDGMENT: "REVELATION_EPISTEMIC",
  INFORMS: "MEDIATION_TRANSFER", MEDIATES: "MEDIATION_TRANSFER",
  TRANSFERS: "MEDIATION_TRANSFER", DELEGATES: "MEDIATION_TRANSFER",
  FINANCIAL_NEED: "MEDIATION_TRANSFER", CULTURAL_EDUCATION: "MEDIATION_TRANSFER",
  DECISION_MAKING: "MEDIATION_TRANSFER",
  CONTRASTS: "THEMATIC_CONTRAST", MIRRORS: "THEMATIC_CONTRAST",
  EXPLAINS: "THEMATIC_EXPLANATION", SUPPORTS: "THEMATIC_EXPLANATION",
  NARRATIVE_COMPOSITE: "THEMATIC_EXPLANATION",
};

if (typeof cytoscape !== "undefined") {
  if (typeof cytoscapeFcose !== "undefined") cytoscape.use(cytoscapeFcose);
  if (typeof cytoscapeCola !== "undefined") cytoscape.use(cytoscapeCola);
}

const TUNING_DEFAULTS = {
  nodeSize: 1.0, edgeWidth: 1.0, arrowScale: 0.8, labelSize: 8, scaleOnZoom: true,
  repulsion: 1.0, edgeLength: 1.0, edgeStrength: 1.0, gravity: 1.0,
  componentSpacing: 1.5, friction: 0.7,
};

const state = {
  manifest: null,
  novelKey: null,
  events: [],
  eventById: new Map(),
  causalEdges: [],
  liveLayout: null,
  tuning: { ...TUNING_DEFAULTS },
  thematicEdges: [],
  agentToEvents: new Map(),
  themeToEvents: { POWER: new Set(), WEALTH: new Set(), KINSHIP: new Set(), JUSTICE: new Set(), KNOWLEDGE: new Set() },
  chapterMin: 1,
  chapterMax: 1,
  cy: null,
  selectedEventId: null,
  focusEventId: null,
  hiddenSupertypes: new Set(),
  hiddenThematicThemes: new Set(),
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
  focusSection: document.getElementById("focus-section"),
  focusLabel: document.getElementById("focus-label"),
  focusClear: document.getElementById("focus-clear"),
  isolateBtn: document.getElementById("isolate-btn"),
  subplotSection: document.getElementById("subplot-section"),
  subplotThemeChecks: document.getElementById("subplot-theme-checks"),
  subplotInfo: document.getElementById("subplot-info"),
  chapterMode: document.getElementById("chapter-mode"),
  chapterRangeRow: document.getElementById("chapter-range-row"),
  chapterSpecificRow: document.getElementById("chapter-specific-row"),
  chapterMin: document.getElementById("chapter-min"),
  chapterMax: document.getElementById("chapter-max"),
  chapterSpecific: document.getElementById("chapter-specific"),
  attrCharacter: document.getElementById("attr-character"),
  attrCharacterList: document.getElementById("attr-character-list"),
  agentSelect: document.getElementById("agent-select"),
  showCausal: document.getElementById("show-causal"),
  showThematic: document.getElementById("show-thematic"),
  showChrono: document.getElementById("show-chrono"),
  hideIsolated: document.getElementById("hide-isolated"),
  thematicThemeFilters: document.getElementById("thematic-theme-filters"),
  edgeConfidence: document.getElementById("edge-confidence"),
  edgeConfidenceVal: document.getElementById("edge-confidence-val"),
  supertypeLegend: document.getElementById("supertype-legend"),
  tNodeSize: document.getElementById("t-node-size"),
  tNodeSizeV: document.getElementById("t-node-size-v"),
  tEdgeWidth: document.getElementById("t-edge-width"),
  tEdgeWidthV: document.getElementById("t-edge-width-v"),
  tArrowScale: document.getElementById("t-arrow-scale"),
  tArrowScaleV: document.getElementById("t-arrow-scale-v"),
  tLabelSize: document.getElementById("t-label-size"),
  tLabelSizeV: document.getElementById("t-label-size-v"),
  tScaleOnZoom: document.getElementById("t-scale-on-zoom"),
  tRepulsion: document.getElementById("t-repulsion"),
  tRepulsionV: document.getElementById("t-repulsion-v"),
  tEdgeLength: document.getElementById("t-edge-length"),
  tEdgeLengthV: document.getElementById("t-edge-length-v"),
  tEdgeStrength: document.getElementById("t-edge-strength"),
  tEdgeStrengthV: document.getElementById("t-edge-strength-v"),
  tGravity: document.getElementById("t-gravity"),
  tGravityV: document.getElementById("t-gravity-v"),
  tComponentSpacing: document.getElementById("t-component-spacing"),
  tComponentSpacingV: document.getElementById("t-component-spacing-v"),
  tFriction: document.getElementById("t-friction"),
  tFrictionV: document.getElementById("t-friction-v"),
  tReset: document.getElementById("t-reset"),
  layoutSelect: document.getElementById("layout-select"),
  reLayout: document.getElementById("re-layout"),
  fitView: document.getElementById("fit-view"),
  maxEvents: document.getElementById("max-events"),
  graph: document.getElementById("graph"),
  detailEmpty: document.getElementById("detail-empty"),
  detailContent: document.getElementById("detail-content"),
  tabDetail: document.getElementById("tab-detail"),
  tabNodes: document.getElementById("tab-nodes"),
  tabCausal: document.getElementById("tab-causal"),
  causalTabEmpty: document.getElementById("causal-tab-empty"),
  causalTabContent: document.getElementById("causal-tab-content"),
  tabButtons: document.querySelectorAll(".tab-btn"),
  nodeList: document.getElementById("node-list"),
  nodeListCount: document.getElementById("node-list-count"),
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

  state.focusEventId = null;
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
  // Supertype legend
  const presentSupertypes = new Map();
  for (const e of state.causalEdges) {
    const st = e.edgeSupertype || "OTHER";
    presentSupertypes.set(st, (presentSupertypes.get(st) || 0) + 1);
  }
  for (const st of [...state.hiddenSupertypes]) {
    if (!presentSupertypes.has(st)) state.hiddenSupertypes.delete(st);
  }
  const legendEntries = [...presentSupertypes.entries()].sort((a, b) => b[1] - a[1]);
  ui.supertypeLegend.innerHTML = "";
  for (const [st, count] of legendEntries) {
    const color = SUPERTYPE_COLOR[st] || SUPERTYPE_DEFAULT;
    const row = document.createElement("div");
    row.className = "legend-row" + (state.hiddenSupertypes.has(st) ? " disabled" : "");
    row.dataset.supertype = st;
    row.title = "Click to toggle";
    row.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>
                     <span class="legend-label">${escapeHtml(st)}</span>
                     <span class="hint">(${count.toLocaleString()})</span>`;
    row.addEventListener("click", () => {
      if (state.hiddenSupertypes.has(st)) state.hiddenSupertypes.delete(st);
      else state.hiddenSupertypes.add(st);
      row.classList.toggle("disabled");
      applyFilters();
    });
    ui.supertypeLegend.appendChild(row);
  }
  if (state.thematicEdges.length) {
    const row = document.createElement("div");
    row.className = "legend-row dashed uninteractive";
    row.style.marginTop = "6px";
    row.innerHTML = `<span class="legend-swatch" style="background:#888"></span>
                     <span class="legend-label">THEMATIC (dashed)</span>
                     <span class="hint">(${state.thematicEdges.length.toLocaleString()})</span>`;
    ui.supertypeLegend.appendChild(row);
  }
  {
    const row = document.createElement("div");
    row.className = "legend-row uninteractive";
    row.innerHTML = `<span class="legend-swatch" style="background:${CHRONO_COLOR}"></span>
                     <span class="legend-label">CHRONOLOGICAL (dotted)</span>
                     <span class="hint">narrative order</span>`;
    ui.supertypeLegend.appendChild(row);
  }

  buildThematicThemeFilters();

  // Node coloring theme checkboxes
  ui.themeFilters.innerHTML = "";
  for (const t of THEMES) {
    const lbl = document.createElement("label");
    lbl.innerHTML = `<input type="checkbox" data-theme="${t}" checked> <span class="theme-${t.toLowerCase()}">${t}</span> <span class="hint">(${state.themeToEvents[t].size})</span>`;
    lbl.querySelector("input").addEventListener("change", applyFilters);
    ui.themeFilters.appendChild(lbl);
  }

  // Subplot multi-theme checkboxes
  ui.subplotThemeChecks.innerHTML = "";
  for (const t of THEMES) {
    const lbl = document.createElement("label");
    lbl.innerHTML = `<input type="checkbox" data-theme="${t}" checked> <span class="theme-${t.toLowerCase()}">${t}</span> <span class="hint">(${state.themeToEvents[t].size})</span>`;
    lbl.querySelector("input").addEventListener("change", () => {
      updateSubplotInfo();
      applyFilters();
    });
    ui.subplotThemeChecks.appendChild(lbl);
  }

  // Chapter range defaults
  ui.chapterMin.value = state.chapterMin;
  ui.chapterMin.min = state.chapterMin;
  ui.chapterMin.max = state.chapterMax;
  ui.chapterMax.value = Math.min(state.chapterMin + 2, state.chapterMax);
  ui.chapterMax.min = state.chapterMin;
  ui.chapterMax.max = state.chapterMax;

  // Attribute search: populate datalist from all characters
  ui.attrCharacterList.innerHTML = "";
  const allChars = [...state.agentToEvents.keys()].sort();
  for (const c of allChars) {
    const opt = document.createElement("option");
    opt.value = c;
    ui.attrCharacterList.appendChild(opt);
  }

  // Legacy agent select (hidden section, kept for backward compat)
  const sortedAgents = [...state.agentToEvents.entries()].sort((a, b) => b[1].size - a[1].size);
  ui.agentSelect.innerHTML = '<option value="">— any —</option>';
  for (const [agent, evs] of sortedAgents) {
    const opt = document.createElement("option");
    opt.value = agent;
    opt.textContent = `${agent} (${evs.size})`;
    ui.agentSelect.appendChild(opt);
  }
}

function buildThematicThemeFilters() {
  const counts = new Map();
  for (const e of state.thematicEdges) {
    if (!e.theme) continue;
    counts.set(e.theme, (counts.get(e.theme) || 0) + 1);
  }
  for (const t of [...state.hiddenThematicThemes]) {
    if (!counts.has(t)) state.hiddenThematicThemes.delete(t);
  }
  ui.thematicThemeFilters.innerHTML = "";
  if (counts.size === 0) {
    ui.thematicThemeFilters.hidden = true;
    return;
  }
  ui.thematicThemeFilters.hidden = false;
  for (const t of THEMES) {
    if (!counts.has(t)) continue;
    const count = counts.get(t);
    const checked = !state.hiddenThematicThemes.has(t);
    const lbl = document.createElement("label");
    lbl.innerHTML = `<input type="checkbox" data-thematic-theme="${t}" ${checked ? "checked" : ""}>
                     <span class="theme-${t.toLowerCase()}">${t}</span>
                     <span class="hint">(${count.toLocaleString()})</span>`;
    lbl.querySelector("input").addEventListener("change", e => {
      if (e.target.checked) state.hiddenThematicThemes.delete(t);
      else state.hiddenThematicThemes.add(t);
      applyFilters();
    });
    ui.thematicThemeFilters.appendChild(lbl);
  }
  syncThematicThemeFiltersDisabled();
}

function syncThematicThemeFiltersDisabled() {
  if (ui.showThematic.checked) ui.thematicThemeFilters.classList.remove("disabled");
  else ui.thematicThemeFilters.classList.add("disabled");
}

function updateSubplotInfo() {
  const checked = [...ui.subplotThemeChecks.querySelectorAll("input[type=checkbox]")]
    .filter(cb => cb.checked).map(cb => cb.dataset.theme);
  if (checked.length === 0) {
    ui.subplotInfo.textContent = "No themes selected — nothing will show.";
  } else if (checked.length === THEMES.length) {
    ui.subplotInfo.textContent = "Showing events with any theme involvement.";
  } else {
    ui.subplotInfo.textContent = `Events with ${checked.join(" or ")} involvement.`;
  }
}

// ---------- Event listeners ----------

ui.themeConfidence.addEventListener("input", () => {
  ui.themeConfidenceVal.textContent = parseFloat(ui.themeConfidence.value).toFixed(2);
  applyFilters();
});

ui.viewMode.addEventListener("change", () => {
  const v = ui.viewMode.value;
  ui.subplotSection.hidden = v !== "subplot";
  ui.focusSection.hidden = v !== "focus";
  if (v !== "focus") {
    state.focusEventId = null;
    updateFocusLabel();
  }
  if (v === "subplot") updateSubplotInfo();
  applyFilters();
});

ui.subplotThemeChecks.addEventListener && (() => {})(); // checkboxes wired in buildSidebar

ui.chapterMode.addEventListener("change", () => {
  const specific = ui.chapterMode.value === "specific";
  ui.chapterRangeRow.hidden = specific;
  ui.chapterSpecificRow.hidden = !specific;
  applyFilters();
});

ui.chapterMin.addEventListener("change", applyFilters);
ui.chapterMax.addEventListener("change", applyFilters);
ui.chapterSpecific.addEventListener("input", debounce(applyFilters, 300));

ui.attrCharacter.addEventListener("input", debounce(applyFilters, 250));
document.querySelectorAll("input[name='attr-role']").forEach(r => r.addEventListener("change", applyFilters));

ui.agentSelect.addEventListener("change", applyFilters);
ui.showCausal.addEventListener("change", applyFilters);
ui.showThematic.addEventListener("change", () => {
  syncThematicThemeFiltersDisabled();
  applyFilters();
});
ui.showChrono.addEventListener("change", applyFilters);
ui.hideIsolated.addEventListener("change", applyFilters);
ui.edgeConfidence.addEventListener("input", () => {
  ui.edgeConfidenceVal.textContent = parseFloat(ui.edgeConfidence.value).toFixed(2);
  applyFilters();
});
ui.maxEvents.addEventListener("change", applyFilters);
ui.search.addEventListener("input", debounce(applyFilters, 250));
ui.layoutSelect.addEventListener("change", () => runLayout());
ui.reLayout.addEventListener("click", () => runLayout());
ui.fitView.addEventListener("click", () => state.cy && state.cy.fit(null, 30));

// "Isolate in graph" button in detail panel
ui.isolateBtn.addEventListener("click", () => {
  if (!state.selectedEventId) return;
  ui.viewMode.value = "focus";
  ui.subplotSection.hidden = true;
  ui.focusSection.hidden = false;
  state.focusEventId = state.selectedEventId;
  updateFocusLabel();
  applyFilters();
});

// Clear focus button
ui.focusClear.addEventListener("click", () => {
  state.focusEventId = null;
  updateFocusLabel();
  applyFilters();
});

// --- Layout-tuning sliders ---

function bindVisualSlider(input, label, key, fmt = v => parseFloat(v).toFixed(1)) {
  input.addEventListener("input", () => {
    state.tuning[key] = parseFloat(input.value);
    label.textContent = fmt(input.value);
    applyVisualTuning();
  });
}
function bindForceSlider(input, label, key, fmt = v => parseFloat(v).toFixed(1) + "×") {
  input.addEventListener("change", () => {
    state.tuning[key] = parseFloat(input.value);
    label.textContent = fmt(input.value);
    runLayout();
  });
  input.addEventListener("input", () => { label.textContent = fmt(input.value); });
}

bindVisualSlider(ui.tNodeSize, ui.tNodeSizeV, "nodeSize");
bindVisualSlider(ui.tEdgeWidth, ui.tEdgeWidthV, "edgeWidth");
bindVisualSlider(ui.tArrowScale, ui.tArrowScaleV, "arrowScale");
bindVisualSlider(ui.tLabelSize, ui.tLabelSizeV, "labelSize", v => String(parseInt(v, 10)));
ui.tScaleOnZoom.addEventListener("change", () => { state.tuning.scaleOnZoom = ui.tScaleOnZoom.checked; applyVisualTuning(); });

bindForceSlider(ui.tRepulsion, ui.tRepulsionV, "repulsion");
bindForceSlider(ui.tEdgeLength, ui.tEdgeLengthV, "edgeLength");
bindForceSlider(ui.tEdgeStrength, ui.tEdgeStrengthV, "edgeStrength", v => parseFloat(v).toFixed(2));
bindForceSlider(ui.tGravity, ui.tGravityV, "gravity");
bindForceSlider(ui.tComponentSpacing, ui.tComponentSpacingV, "componentSpacing");
bindForceSlider(ui.tFriction, ui.tFrictionV, "friction", v => parseFloat(v).toFixed(2));

ui.tReset.addEventListener("click", () => {
  state.tuning = { ...TUNING_DEFAULTS };
  ui.tNodeSize.value = TUNING_DEFAULTS.nodeSize; ui.tNodeSizeV.textContent = TUNING_DEFAULTS.nodeSize.toFixed(1);
  ui.tEdgeWidth.value = TUNING_DEFAULTS.edgeWidth; ui.tEdgeWidthV.textContent = TUNING_DEFAULTS.edgeWidth.toFixed(1);
  ui.tArrowScale.value = TUNING_DEFAULTS.arrowScale; ui.tArrowScaleV.textContent = TUNING_DEFAULTS.arrowScale.toFixed(1);
  ui.tLabelSize.value = TUNING_DEFAULTS.labelSize; ui.tLabelSizeV.textContent = String(TUNING_DEFAULTS.labelSize);
  ui.tScaleOnZoom.checked = TUNING_DEFAULTS.scaleOnZoom;
  ui.tRepulsion.value = TUNING_DEFAULTS.repulsion; ui.tRepulsionV.textContent = TUNING_DEFAULTS.repulsion.toFixed(1) + "×";
  ui.tEdgeLength.value = TUNING_DEFAULTS.edgeLength; ui.tEdgeLengthV.textContent = TUNING_DEFAULTS.edgeLength.toFixed(1) + "×";
  ui.tEdgeStrength.value = TUNING_DEFAULTS.edgeStrength; ui.tEdgeStrengthV.textContent = TUNING_DEFAULTS.edgeStrength.toFixed(2);
  ui.tGravity.value = TUNING_DEFAULTS.gravity; ui.tGravityV.textContent = TUNING_DEFAULTS.gravity.toFixed(1) + "×";
  ui.tComponentSpacing.value = TUNING_DEFAULTS.componentSpacing; ui.tComponentSpacingV.textContent = TUNING_DEFAULTS.componentSpacing.toFixed(1) + "×";
  ui.tFriction.value = TUNING_DEFAULTS.friction; ui.tFrictionV.textContent = TUNING_DEFAULTS.friction.toFixed(2);
  applyVisualTuning();
  runLayout();
});

function debounce(fn, ms) {
  let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---------- Filter / render ----------

function parseChapterSpec(spec) {
  const chapters = new Set();
  for (const part of spec.split(",")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    const range = trimmed.split("-");
    if (range.length === 2) {
      const from = parseInt(range[0], 10);
      const to = parseInt(range[1], 10);
      if (!isNaN(from) && !isNaN(to)) {
        for (let i = Math.min(from, to); i <= Math.max(from, to); i++) chapters.add(i);
      }
    } else {
      const n = parseInt(trimmed, 10);
      if (!isNaN(n)) chapters.add(n);
    }
  }
  return chapters;
}

function applyFilters() {
  const themesEnabled = new Set(
    [...ui.themeFilters.querySelectorAll("input[type=checkbox]")]
      .filter(cb => cb.checked).map(cb => cb.dataset.theme)
  );
  const minConf = parseFloat(ui.themeConfidence.value);

  // Chapter filter
  const chMode = ui.chapterMode.value;
  const chMin = parseInt(ui.chapterMin.value, 10);
  const chMax = parseInt(ui.chapterMax.value, 10);
  const specificChapters = chMode === "specific" ? parseChapterSpec(ui.chapterSpecific.value) : null;

  // Attribute search
  const attrChar = ui.attrCharacter.value.trim().toLowerCase();
  const attrRole = (document.querySelector("input[name='attr-role']:checked") || {}).value || "any";

  const search = ui.search.value.trim().toLowerCase();
  const view = ui.viewMode.value;

  // Subplot: multiple themes (checkboxes)
  const subplotThemes = new Set(
    [...ui.subplotThemeChecks.querySelectorAll("input[type=checkbox]")]
      .filter(cb => cb.checked).map(cb => cb.dataset.theme)
  );

  const showCausal = ui.showCausal.checked;
  const showThematic = ui.showThematic.checked;
  const showChrono = ui.showChrono.checked;
  const hideIsolated = ui.hideIsolated.checked;
  const minEdgeConf = parseFloat(ui.edgeConfidence.value);
  const maxEvents = parseInt(ui.maxEvents.value, 10);

  // Focus mode: build 1-hop causal neighborhood when an event is isolated.
  // Bypasses chapter filter to show all connections across the full novel.
  let focusNeighborhood = null;
  if (view === "focus" && state.focusEventId) {
    focusNeighborhood = new Set([state.focusEventId]);
    for (const e of state.causalEdges) {
      if (e.from === state.focusEventId) focusNeighborhood.add(e.to);
      if (e.to === state.focusEventId) focusNeighborhood.add(e.from);
    }
  }

  // Event filter
  let visible = new Set();
  for (const ev of state.events) {
    if (view === "focus" && focusNeighborhood) {
      // Isolated mode: only show the neighborhood, skip chapter filter
      if (!focusNeighborhood.has(ev.id)) continue;
    } else {
      // All other modes (including focus with no event selected): use chapter filter
      if (chMode === "specific") {
        if (!specificChapters || !specificChapters.has(ev.chapter)) continue;
      } else {
        if (ev.chapter < chMin || ev.chapter > chMax) continue;
      }
    }

    // Attribute/role filter
    if (attrChar) {
      const isActor = ev.actors.some(a => a.toLowerCase().includes(attrChar));
      const isPatient = ev.patients.some(p => p.toLowerCase().includes(attrChar));
      if (attrRole === "actor" && !isActor) continue;
      else if (attrRole === "patient" && !isPatient) continue;
      else if (attrRole === "any" && !isActor && !isPatient) continue;
    }

    if (search && !ev.description.toLowerCase().includes(search) && !ev.sourceQuote.toLowerCase().includes(search)) continue;

    // Subplot: filter to events matching any selected theme
    if (view === "subplot" && subplotThemes.size > 0) {
      const hasTheme = [...subplotThemes].some(t => {
        const td = ev.themes[t];
        if (!td) return false;
        if (td.involvement !== "direct" && td.involvement !== "indirect") return false;
        return (td.confidence ?? 0) >= minConf;
      });
      if (!hasTheme) continue;
    } else if (view === "subplot" && subplotThemes.size === 0) {
      continue; // no themes checked = show nothing
    }

    visible.add(ev.id);
    if (visible.size >= maxEvents) break;
  }

  // Edge filter
  const causalShown = !showCausal ? [] : state.causalEdges.filter(e => {
    if (!visible.has(e.from) || !visible.has(e.to)) return false;
    if ((e.confidence ?? 0) < minEdgeConf) return false;
    const st = e.edgeSupertype || "OTHER";
    if (state.hiddenSupertypes.has(st)) return false;
    return true;
  });
  const thematicFiltered = !showThematic ? [] : state.thematicEdges.filter(e => {
    if (!visible.has(e.from) || !visible.has(e.to)) return false;
    if (view === "subplot" && subplotThemes.size > 0 && !subplotThemes.has(e.theme)) return false;
    if ((e.confidence ?? 0) < minEdgeConf) return false;
    if (e.theme && state.hiddenThematicThemes.has(e.theme)) return false;
    return true;
  });

  const finalCausal = view === "subplot" ? [] : causalShown;
  const finalThematic = view === "subplot" || showThematic ? thematicFiltered : [];

  function buildChrono(set) {
    if (!showChrono || set.size < 2) return [];
    const ordered = [...set]
      .map(id => state.eventById.get(id)).filter(Boolean)
      .sort((a, b) => (a.chapter - b.chapter) || (a.sequence - b.sequence));
    const out = [];
    for (let i = 0; i < ordered.length - 1; i++) {
      const a = ordered[i], b = ordered[i + 1];
      out.push({ id: `chrono:${a.id}->${b.id}`, from: a.id, to: b.id, confidence: 1.0 });
    }
    return out;
  }
  let chronoEdges = buildChrono(visible);

  if (hideIsolated) {
    const connected = new Set();
    for (const e of finalCausal) { connected.add(e.from); connected.add(e.to); }
    for (const e of finalThematic) { connected.add(e.from); connected.add(e.to); }
    for (const e of chronoEdges) { connected.add(e.from); connected.add(e.to); }
    const before = visible.size;
    visible = new Set([...visible].filter(id => connected.has(id)));
    if (visible.size !== before) chronoEdges = buildChrono(visible);
  }

  ui.statShown.textContent = `shown: ${visible.size.toLocaleString()} / ${state.events.length.toLocaleString()}`;

  if (view === "subplot") {
    const checked = [...subplotThemes];
    if (checked.length === 0) {
      ui.subplotInfo.textContent = "No themes selected.";
    } else {
      const total = [...visible].reduce((acc, id) => {
        const ev = state.eventById.get(id);
        return ev ? acc + 1 : acc;
      }, 0);
      ui.subplotInfo.textContent = `${total} events shown with ${checked.join(" or ")} involvement.`;
    }
  }

  if (view === "focus" && !state.focusEventId) {
    ui.statShown.textContent += " — click a node, then Isolate";
  }

  render(visible, finalCausal, finalThematic, chronoEdges, themesEnabled, minConf);
  updateNodeList(visible);
}

function updateNodeList(visibleSet) {
  const ordered = [...visibleSet]
    .map(id => state.eventById.get(id)).filter(Boolean)
    .sort((a, b) => (a.chapter - b.chapter) || (a.sequence - b.sequence));
  ui.nodeListCount.textContent = ordered.length ? `(${ordered.length.toLocaleString()})` : "";
  ui.nodeList.innerHTML = "";
  for (const ev of ordered) {
    const row = document.createElement("div");
    row.className = "node-list-item" + (ev.id === state.selectedEventId ? " selected" : "");
    row.dataset.id = ev.id;
    row.innerHTML = `<span class="seq">${ev.chapter}.${ev.sequence}</span>
                     <span class="desc" title="${escapeHtml(ev.description)}">${escapeHtml(truncate(ev.description, 120))}</span>`;
    row.addEventListener("click", () => {
      if (state.cy && state.cy.getElementById(ev.id).length) {
        state.cy.center(state.cy.getElementById(ev.id));
      }
      showDetail(ev.id);
      switchTab("detail");
    });
    ui.nodeList.appendChild(row);
  }
}

function switchTab(name) {
  ui.tabButtons.forEach(btn => btn.classList.toggle("active", btn.dataset.tab === name));
  ui.tabDetail.hidden = name !== "detail";
  ui.tabNodes.hidden = name !== "nodes";
  ui.tabCausal.hidden = name !== "causal";
}

ui.tabButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    switchTab(btn.dataset.tab);
    // If switching to Causal tab and a node is already selected, rebuild it
    if (btn.dataset.tab === "causal" && state.selectedEventId) {
      const ev = state.eventById.get(state.selectedEventId);
      if (ev) buildCausalTab(ev);
    }
  });
});

function render(visibleSet, causalEdges, thematicEdges, chronoEdges, themesEnabled, minConf) {
  if (visibleSet.size === 0) {
    if (state.cy) state.cy.elements().remove();
    return;
  }

  const nodes = [];
  for (const id of visibleSet) {
    const ev = state.eventById.get(id);
    nodes.push({
      data: {
        id: ev.id,
        label: `${ev.chapter}.${ev.sequence}`,
        fullId: ev.id,
        chapter: ev.chapter,
        seq: `${ev.chapter}.${ev.sequence}`,
        description: ev.description,
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
  for (const e of chronoEdges) {
    edges.push({
      data: {
        id: e.id, source: e.from, target: e.to,
        relType: "CHRONOLOGICAL", kind: "chrono",
        confidence: 1.0, edgeColor: CHRONO_COLOR, weight: 0.15,
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
    state.cy.on("tap", "node", evt => {
      const id = evt.target.id();
      showDetail(id);
      switchTab("detail");
      // In focus mode (when already isolated), clicking a neighbor re-isolates on it
      if (ui.viewMode.value === "focus" && state.focusEventId) {
        state.focusEventId = id;
        updateFocusLabel();
        applyFilters();
      }
    });
    state.cy.on("tap", "edge", evt => { showEdgeDetail(evt.target.data()); switchTab("detail"); });
  } else {
    state.cy.elements().remove();
    state.cy.add(elements);
  }
  runLayout();
}

function cyStyle() {
  const t = state.tuning;
  const baseSize = 16 * t.nodeSize;
  const selSize = 22 * t.nodeSize;
  return [
    {
      selector: "node",
      style: {
        "background-color": "data(themeColor)",
        "label": "data(label)",
        "color": "#fff",
        "font-size": t.labelSize,
        "min-zoomed-font-size": t.scaleOnZoom ? 0 : t.labelSize,
        "text-valign": "bottom",
        "text-halign": "center",
        "text-margin-y": 3,
        "width": baseSize, "height": baseSize,
        "border-color": "#0a0c12", "border-width": 1,
      },
    },
    {
      selector: "node:selected",
      style: { "border-color": "#fff", "border-width": 3, "width": selSize, "height": selSize },
    },
    {
      selector: "edge[kind = 'causal']",
      style: {
        "width": `mapData(confidence, 0.4, 0.9, ${0.6 * t.edgeWidth}, ${5 * t.edgeWidth})`,
        "opacity": "mapData(confidence, 0.4, 0.9, 0.35, 1)",
        "line-color": "data(edgeColor)",
        "target-arrow-color": "data(edgeColor)",
        "target-arrow-shape": "triangle",
        "arrow-scale": t.arrowScale,
        "curve-style": "bezier",
      },
    },
    {
      selector: "edge[kind = 'thematic']",
      style: {
        "width": `mapData(confidence, 0.3, 0.9, ${0.8 * t.edgeWidth}, ${5.5 * t.edgeWidth})`,
        "opacity": "mapData(confidence, 0.3, 0.9, 0.4, 1)",
        "line-color": "data(edgeColor)",
        "target-arrow-color": "data(edgeColor)",
        "target-arrow-shape": "triangle",
        "arrow-scale": t.arrowScale,
        "line-style": "dashed",
        "curve-style": "bezier",
      },
    },
    {
      selector: "edge[kind = 'chrono']",
      style: {
        "width": Math.max(0.8, 1.2 * t.edgeWidth),
        "opacity": 0.55,
        "line-color": "data(edgeColor)",
        "target-arrow-color": "data(edgeColor)",
        "target-arrow-shape": "triangle",
        "arrow-scale": t.arrowScale * 0.8,
        "line-style": "dotted",
        "curve-style": "bezier",
        "z-index": 1,
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
  if (state.liveLayout) {
    try { state.liveLayout.stop(); } catch (e) {}
    state.liveLayout = null;
  }
  const t = state.tuning;
  const name = ui.layoutSelect.value;
  const opts = {
    fcose: {
      name: "fcose", animate: true, animationDuration: 600, randomize: true, quality: "default",
      nodeRepulsion: 8000 * t.repulsion,
      idealEdgeLength: edge => (80 + (1 - (edge.data("confidence") || 0.5)) * 220) * t.edgeLength,
      edgeElasticity: 0.45 * t.edgeStrength,
      gravity: 0.18 * t.gravity,
      gravityRangeCompound: 1.5 * t.componentSpacing,
      nodeSeparation: 80 * t.componentSpacing,
      packComponents: true, padding: 40,
    },
    "cola-live": {
      name: "cola", animate: true, infinite: true, fit: false,
      randomize: false, avoidOverlap: true, handleDisconnected: true,
      nodeSpacing: 18 * t.componentSpacing,
      edgeLength: edge => (70 + (1 - (edge.data("confidence") || 0.5)) * 200) * t.edgeLength,
      edgeSymDiffLength: 30,
      flow: { axis: "y", minSeparation: 0 },
      maxSimulationTime: 4000,
      convergenceThreshold: 0.001 * (1 - t.friction),
    },
    cose: {
      name: "cose", animate: "end", animationDuration: 600,
      idealEdgeLength: 220 * t.edgeLength,
      nodeRepulsion: 800000 * t.repulsion,
      nodeOverlap: 24, gravity: 30 * t.gravity, numIter: 1500, padding: 40,
      edgeElasticity: 100 * t.edgeStrength,
    },
    dagre: { name: "dagre", rankDir: "LR", nodeSep: 30, rankSep: 90 * t.edgeLength, animate: true, animationDuration: 500, padding: 30 },
    grid: { name: "grid", padding: 30, animate: true, animationDuration: 400, avoidOverlap: true },
    concentric: {
      name: "concentric",
      concentric: n => -n.data("chapter"),
      levelWidth: () => 1,
      minNodeSpacing: 18 * t.componentSpacing,
      animate: true, animationDuration: 500, padding: 30,
    },
  };
  const cfg = opts[name] || opts.fcose;
  const layout = state.cy.layout(cfg);
  layout.run();
  if (cfg.infinite) state.liveLayout = layout;
}

function applyVisualTuning() {
  if (!state.cy) return;
  state.cy.style(cyStyle());
}

function dominantThemeColor(ev, themesEnabled, minConf) {
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
  // Show Isolate button whenever a node is selected
  ui.isolateBtn.hidden = false;
  ui.isolateBtn.textContent = state.focusEventId === eventId ? "⬡ Already isolated" : "⬡ Isolate in graph";

  if (ui.nodeList) {
    ui.nodeList.querySelectorAll(".node-list-item").forEach(el => {
      el.classList.toggle("selected", el.dataset.id === eventId);
    });
  }

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
  const shortId = ev.id.includes("/") ? ev.id.split("/").pop() : ev.id;

  ui.detailContent.innerHTML = `
    <div class="event-id-label">${escapeHtml(ev.id)}</div>
    <h2>${escapeHtml(truncate(ev.description, 90))}</h2>
    <div class="meta-row">Ch ${ev.chapter} · event #${ev.sequence} · ${escapeHtml(ev.actionType)} · conf ${ev.confidence.toFixed(2)}</div>
    ${ev.sourceQuote
      ? `<div class="detail-section"><div class="panel-title">Actual text</div><div class="quote">${escapeHtml(ev.sourceQuote)}</div></div>`
      : ""}

    <div class="detail-section">
      <div class="panel-title">Themes</div>
      <table class="themes">${themeRows}</table>
    </div>

    ${ev.actors.length ? `
    <div class="detail-section">
      <div class="panel-title">Actors</div>
      <div class="chips">${ev.actors.map(a => `<span class="chip" data-agent="${escapeHtml(a)}" data-role="actor">${escapeHtml(a)}</span>`).join("")}</div>
    </div>` : ""}

    ${ev.patients.length ? `
    <div class="detail-section">
      <div class="panel-title">Patients</div>
      <div class="chips">${ev.patients.map(a => `<span class="chip" data-agent="${escapeHtml(a)}" data-role="patient">${escapeHtml(a)}</span>`).join("")}</div>
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
      ${causalIn.length > 8 ? `<div class="hint">+${causalIn.length - 8} more — see Causal tab</div>` : ""}
    </div>

    <div class="detail-section">
      <div class="panel-title">Causes (${causalOut.length})</div>
      ${causalOut.slice(0, 8).map(e => {
        const dst = state.eventById.get(e.to);
        return `<div class="neighbor" data-jump="${e.to}"><span class="rel">${escapeHtml(e.relationType)}</span> ${escapeHtml(truncate(dst ? dst.description : e.to, 70))}</div>`;
      }).join("") || `<div class="hint">none in current data</div>`}
      ${causalOut.length > 8 ? `<div class="hint">+${causalOut.length - 8} more — see Causal tab</div>` : ""}
    </div>
  `;

  ui.detailContent.querySelectorAll("[data-jump]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.jump;
      if (state.cy && state.cy.getElementById(id).length) state.cy.center(state.cy.getElementById(id));
      showDetail(id);
    });
  });

  // Clicking an actor/patient chip filters the attribute search by role
  ui.detailContent.querySelectorAll("[data-agent]").forEach(el => {
    el.addEventListener("click", () => {
      ui.attrCharacter.value = el.dataset.agent;
      const role = el.dataset.role; // "actor" or "patient"
      const radio = document.querySelector(`input[name='attr-role'][value='${role}']`);
      if (radio) radio.checked = true;
      applyFilters();
    });
  });

  // Populate causal tab content
  buildCausalTab(ev);
}

function buildCausalTab(ev) {
  const causalIn = state.causalEdges.filter(e => e.to === ev.id);
  const causalOut = state.causalEdges.filter(e => e.from === ev.id);

  ui.causalTabEmpty.hidden = true;
  ui.causalTabContent.hidden = false;

  const makeRow = (e, neighborId) => {
    const neighbor = state.eventById.get(neighborId);
    const shortId = neighborId.includes("/") ? neighborId.split("/").pop() : neighborId;
    return `<div class="neighbor" data-jump="${escapeHtml(neighborId)}">
      <span class="rel">${escapeHtml(e.relationType)}</span>
      <span class="hint" style="font-family:monospace">${escapeHtml(shortId)}</span>
      ${escapeHtml(truncate(neighbor ? neighbor.description : neighborId, 80))}
      ${neighbor ? `<span class="hint"> (Ch ${neighbor.chapter}.${neighbor.sequence})</span>` : ""}
    </div>`;
  };

  ui.causalTabContent.innerHTML = `
    <div class="event-id-label" style="margin-bottom:8px">${escapeHtml(ev.id)}</div>
    <div class="panel-title">Causes → (${causalOut.length} outgoing)</div>
    ${causalOut.map(e => makeRow(e, e.to)).join("") || '<div class="hint">none</div>'}
    <div class="panel-title" style="margin-top:12px">← Caused by (${causalIn.length} incoming)</div>
    ${causalIn.map(e => makeRow(e, e.from)).join("") || '<div class="hint">none</div>'}
  `;

  ui.causalTabContent.querySelectorAll("[data-jump]").forEach(el => {
    el.addEventListener("click", () => {
      const id = el.dataset.jump;
      if (state.cy && state.cy.getElementById(id).length) state.cy.center(state.cy.getElementById(id));
      showDetail(id);
      switchTab("detail");
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
  let kindLabel;
  if (d.kind === "causal") {
    kindLabel = `${colorChip}${escapeHtml(d.relType)}${d.supertype && d.supertype !== "OTHER" ? ` <span class="hint">(${escapeHtml(d.supertype)})</span>` : ""}`;
  } else if (d.kind === "chrono") {
    kindLabel = `${colorChip}CHRONOLOGICAL <span class="hint">(narrative order, overlay)</span>`;
  } else {
    kindLabel = `${colorChip}THEMATIC · ${escapeHtml(d.relType)}`;
  }

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

function updateFocusLabel() {
  if (!state.focusEventId) {
    ui.focusLabel.textContent = "Click any node then use "Isolate in graph".";
    ui.focusClear.hidden = true;
  } else {
    const ev = state.eventById.get(state.focusEventId);
    ui.focusLabel.textContent = ev ? `Focused: ${truncate(ev.description, 60)}` : state.focusEventId;
    ui.focusClear.hidden = false;
  }
}

init();
