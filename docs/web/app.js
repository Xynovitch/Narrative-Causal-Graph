/* CEKG webpage. Loads a pipeline JSON-LD file and renders an interactive view. */

const THEMES = ["POWER", "WEALTH", "KINSHIP", "JUSTICE", "KNOWLEDGE"];
const THEME_COLOR = {
  POWER: "#e15759", WEALTH: "#f1ce63", KINSHIP: "#59a14f",
  JUSTICE: "#8cd17d", KNOWLEDGE: "#4e79a7",
};
const ACTION_COLOR_DEFAULT = "#7a8aa8";

const state = {
  manifest: null,
  novelKey: null,
  events: [],
  eventById: new Map(),
  causalEdges: [],
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
      state.causalEdges.push({
        id: item["@id"],
        from: item.from, to: item.to,
        relationType: item.relationType || "",
        mechanism: item.mechanism || "",
        weight: item.weight ?? 1.0,
        confidence: item.confidence ?? 1.0,
        edgeSupertype: item.edge_supertype || null,
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
  ui.chapterMax.value = Math.min(state.chapterMin + 4, state.chapterMax);
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
  const maxEvents = parseInt(ui.maxEvents.value, 10);

  // Event filter
  let visible = new Set();
  for (const ev of state.events) {
    if (ev.chapter < chMin || ev.chapter > chMax) continue;
    if (agent) {
      if (!ev.actors.includes(agent) && !ev.patients.includes(agent)) continue;
    }
    if (search && !ev.description.toLowerCase().includes(search) && !ev.sourceQuote.toLowerCase().includes(search)) continue;

    if (view === "subplot") {
      const td = ev.themes[subplotTheme];
      if (!td) continue;
      if (td.involvement !== "direct" && td.involvement !== "indirect") continue;
      if ((td.confidence ?? 0) < minConf) continue;
    } else {
      // For other views: must touch at least one enabled theme above threshold,
      // OR have no theme annotations at all (don't hide unannotated events).
      const themeKeys = Object.keys(ev.themes || {});
      if (themeKeys.length > 0) {
        const ok = THEMES.some(t => {
          if (!themesEnabled.has(t)) return false;
          const td = ev.themes[t];
          if (!td) return false;
          if (td.involvement !== "direct" && td.involvement !== "indirect") return false;
          return (td.confidence ?? 0) >= minConf;
        });
        // If themes exist but none match enabled-with-threshold, hide
        // — UNLESS user disabled the theme filter entirely (all unchecked = show all)
        if (!ok && themesEnabled.size > 0) continue;
      }
    }

    visible.add(ev.id);
    if (visible.size >= maxEvents) break;
  }

  // Edge filter — both endpoints must be visible
  const causalShown = !showCausal ? [] : state.causalEdges.filter(e => visible.has(e.from) && visible.has(e.to));
  const thematicShown = (!showThematic || (view === "subplot" && false)) ? state.thematicEdges : state.thematicEdges;
  const thematicFiltered = !showThematic ? [] : thematicShown.filter(e => {
    if (!visible.has(e.from) || !visible.has(e.to)) return false;
    if (view === "subplot" && e.theme !== subplotTheme) return false;
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
  render(visible, finalCausal, finalThematic);
}

function render(visibleSet, causalEdges, thematicEdges) {
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
        themeColor: dominantThemeColor(ev),
      },
    });
  }
  const edges = [];
  for (const e of causalEdges) {
    edges.push({ data: { id: e.id, source: e.from, target: e.to, relType: e.relationType, kind: "causal" } });
  }
  for (const e of thematicEdges) {
    edges.push({ data: { id: e.id, source: e.from, target: e.to, relType: e.theme, kind: "thematic", themeColor: THEME_COLOR[e.theme] || "#888" } });
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
      selector: "edge[kind = 'causal']",
      style: {
        "width": 1, "line-color": "#444a6a", "target-arrow-color": "#666",
        "target-arrow-shape": "triangle", "curve-style": "bezier", "opacity": 0.6,
      },
    },
    {
      selector: "edge[kind = 'thematic']",
      style: {
        "width": 1.5, "line-color": "data(themeColor)",
        "target-arrow-color": "data(themeColor)", "target-arrow-shape": "triangle",
        "curve-style": "bezier", "opacity": 0.7, "line-style": "dashed",
      },
    },
  ];
}

function runLayout() {
  if (!state.cy) return;
  const name = ui.layoutSelect.value;
  const opts = {
    cose: { name: "cose", animate: false, idealEdgeLength: 80, nodeRepulsion: 8000, padding: 20 },
    dagre: { name: "dagre", rankDir: "LR", nodeSep: 20, rankSep: 60, animate: false },
    grid: { name: "grid", padding: 20, animate: false },
    concentric: { name: "concentric", concentric: n => n.data("chapter"), levelWidth: () => 1, animate: false },
  };
  state.cy.layout(opts[name] || opts.cose).run();
}

function dominantThemeColor(ev) {
  let bestT = null, bestC = 0;
  for (const t of THEMES) {
    const td = ev.themes[t];
    if (!td) continue;
    if (td.involvement !== "direct" && td.involvement !== "indirect") continue;
    if ((td.confidence ?? 0) > bestC) { bestC = td.confidence; bestT = t; }
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

init();
