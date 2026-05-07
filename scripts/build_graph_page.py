#!/usr/bin/env python3
"""
Builds a self-contained Cytoscape.js HTML explorer from the neo4j_csv exports.
Run: python3 build_graph_page.py
Output: graph_explorer.html
"""

import csv
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(PROJECT_ROOT, "neo4j_csv")

SUPERTYPE_COLORS = {
    "CAUSAL_PRODUCTION":   "#4e8ef7",
    "NARRATIVE_ESCALATION": "#f7a144",
    "EMOTIONAL_DRIVE":     "#e05c8a",
    "SOCIAL_BOND":         "#4ecf7a",
    "CAUSAL_CONSTRAINT":   "#a44ef7",
}

ACTION_TYPE_COLORS = {
    "CONTEMPLATION":               "#a8c8f0",
    "CHARACTER_EMOTION_SPIKE":     "#f0a8c0",
    "PHYSICAL_ACTION":             "#f0c8a8",
    "PHYSICAL_MOVEMENT":           "#f0e0a8",
    "CONFLICT_IGNITION":           "#f08080",
    "CHARACTER_INTRODUCTION":      "#c0d8a0",
    "ARRIVAL":                     "#d0e8c0",
    "CONNECTION_DEEPENED":         "#b8e0d8",
    "ADVICE_OFFERED":              "#d8d0f0",
    "ACKNOWLEDGMENT":              "#e8e0c0",
    "ACTION_TENSION":              "#f0c0a0",
    "ADAPTATION_TO_CHANGE":        "#c0e0e0",
    "ALLY_GATHERING":              "#b0d0b0",
    "ANTICIPATION":                "#e8d0b0",
    "ASPIRATION_REVEALED":         "#d0b8f0",
    "AUTHOR_INTENT_SIGNAL":        "#e0e0e0",
    "AUTONOMY_ASSERTION":          "#f0d8b0",
    "CHARACTER_TRANSFORMATION_SIGNAL": "#c8b8f0",
    "COMMUNITY_STRESS":            "#f0b8b8",
    "COMPANIONSHIP_FORMED":        "#b8f0d0",
    "COMPARISON_POINT":            "#e0d8c0",
    "COMPETITION_INITIATED":       "#f0c8c0",
}
DEFAULT_NODE_COLOR = "#cccccc"


def load_events():
    nodes = []
    with open(os.path.join(CSV_DIR, "events.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            chapter = row["chapter"]
            try:
                ch_int = int(chapter)
            except ValueError:
                ch_int = 999
            color = ACTION_TYPE_COLORS.get(row["actionType"], DEFAULT_NODE_COLOR)
            nodes.append({
                "data": {
                    "id": row[":ID"],
                    "label": (row["name"][:60] + "…") if len(row["name"]) > 60 else row["name"],
                    "fullName": row["name"],
                    "actionType": row["actionType"],
                    "chapter": ch_int,
                    "confidence": float(row["confidence"]) if row["confidence"] else 0.0,
                    "location": row["location"],
                    "time": row["time"],
                    "scene_id": row["scene_id"],
                    "color": color,
                }
            })
    return nodes


def load_edges():
    edges = []
    with open(os.path.join(CSV_DIR, "causes.csv"), newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            supertype = row["edge_supertype"]
            color = SUPERTYPE_COLORS.get(supertype, "#999999")
            edges.append({
                "data": {
                    "id": f"e{i}",
                    "source": row[":START_ID"],
                    "target": row[":END_ID"],
                    "relationType": row["relationType"],
                    "supertype": supertype,
                    "mechanism": row["mechanism"],
                    "confidence": float(row["confidence"]) if row["confidence"] else 0.0,
                    "color": color,
                }
            })
    return edges


def build_html(nodes, edges):
    elements_json = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)

    max_chapter = max(n["data"]["chapter"] for n in nodes if n["data"]["chapter"] < 999)
    supertypes = sorted(SUPERTYPE_COLORS.keys())
    action_types = sorted(ACTION_TYPE_COLORS.keys())

    supertype_legend = "\n".join(
        f'<div class="legend-item"><span class="dot" style="background:{c}"></span>{s}</div>'
        for s, c in SUPERTYPE_COLORS.items()
    )
    action_legend = "\n".join(
        f'<div class="legend-item"><span class="dot" style="background:{c}"></span>{s}</div>'
        for s, c in sorted(ACTION_TYPE_COLORS.items())
    )
    supertype_options = "\n".join(
        f'<label><input type="checkbox" class="supertype-cb" value="{s}" checked> {s}</label>'
        for s in supertypes
    )
    action_options = "\n".join(
        f'<label><input type="checkbox" class="action-cb" value="{a}" checked> {a}</label>'
        for a in action_types
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Causal Event Knowledge Graph — Great Expectations</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; background: #0f1117; color: #e0e0e0; }}

  #topbar {{
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px; background: #1a1d27; border-bottom: 1px solid #2d3045;
    flex-shrink: 0;
  }}
  #topbar h1 {{ font-size: 14px; font-weight: 600; color: #c0c8f0; white-space: nowrap; }}
  #search {{ flex: 1; max-width: 300px; padding: 5px 10px; border-radius: 6px; border: 1px solid #3a3d55; background: #252840; color: #e0e0e0; font-size: 13px; }}
  #search::placeholder {{ color: #777; }}
  .stat {{ font-size: 12px; color: #888; white-space: nowrap; }}

  #main {{ display: flex; flex: 1; overflow: hidden; }}

  #sidebar {{
    width: 260px; min-width: 220px; background: #1a1d27; border-right: 1px solid #2d3045;
    overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 14px;
    flex-shrink: 0;
  }}
  .panel-title {{ font-size: 11px; font-weight: 700; letter-spacing: .08em; color: #888; text-transform: uppercase; margin-bottom: 6px; }}
  label {{ display: block; font-size: 12px; color: #ccc; margin-bottom: 3px; cursor: pointer; }}
  label input[type=checkbox] {{ margin-right: 5px; accent-color: #6070f0; }}
  input[type=range] {{ width: 100%; accent-color: #6070f0; }}
  .range-val {{ font-size: 11px; color: #aaa; text-align: right; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: #bbb; margin-bottom: 3px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  select {{ width: 100%; padding: 4px 6px; border-radius: 5px; border: 1px solid #3a3d55; background: #252840; color: #e0e0e0; font-size: 12px; }}
  button {{ padding: 5px 10px; border-radius: 5px; border: none; background: #3a4060; color: #c0d0ff; cursor: pointer; font-size: 12px; }}
  button:hover {{ background: #4a5080; }}
  #chapter-display {{ display: flex; justify-content: space-between; font-size: 12px; color: #aaa; }}

  #cy {{ flex: 1; background: #0f1117; }}

  #detail-panel {{
    width: 300px; min-width: 240px; background: #1a1d27; border-left: 1px solid #2d3045;
    overflow-y: auto; padding: 14px; flex-shrink: 0; display: none;
  }}
  #detail-panel.visible {{ display: block; }}
  #detail-title {{ font-size: 13px; font-weight: 600; color: #c8d8ff; margin-bottom: 10px; line-height: 1.4; }}
  .detail-row {{ font-size: 12px; color: #bbb; margin-bottom: 6px; }}
  .detail-row span {{ color: #e0e0e0; }}
  .detail-label {{ color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 2px; }}
  #close-detail {{ float: right; cursor: pointer; color: #888; font-size: 16px; line-height: 1; }}
  #close-detail:hover {{ color: #fff; }}
  #neighbor-list {{ margin-top: 10px; }}
  .neighbor-item {{ font-size: 11px; color: #aaa; padding: 4px 0; border-bottom: 1px solid #2a2d3a; cursor: pointer; }}
  .neighbor-item:hover {{ color: #c0d0ff; }}
</style>
</head>
<body>

<div id="topbar">
  <h1>Causal Event Knowledge Graph &mdash; Great Expectations</h1>
  <input id="search" type="text" placeholder="Search events…">
  <span class="stat" id="visible-stat"></span>
  <button id="fit-btn">Fit view</button>
  <button id="reset-btn">Reset filters</button>
</div>

<div id="main">
  <div id="sidebar">

    <div>
      <div class="panel-title">Layout</div>
      <select id="layout-select">
        <option value="dagre">Dagre (hierarchical)</option>
        <option value="breadthfirst">Breadth-first</option>
        <option value="cose">CoSE (force-directed)</option>
        <option value="grid">Grid</option>
        <option value="circle">Circle</option>
      </select>
      <br><br>
      <button id="apply-layout" style="width:100%">Apply layout</button>
    </div>

    <div>
      <div class="panel-title">Chapter range</div>
      <div id="chapter-display"><span id="ch-lo-label">1</span><span id="ch-hi-label">{max_chapter}</span></div>
      From: <input type="range" id="ch-lo" min="1" max="{max_chapter}" value="1">
      To: <input type="range" id="ch-hi" min="1" max="{max_chapter}" value="5">
    </div>

    <div>
      <div class="panel-title">Min confidence</div>
      <input type="range" id="conf-slider" min="0" max="1" step="0.05" value="0">
      <div class="range-val"><span id="conf-val">0.00</span></div>
    </div>

    <div>
      <div class="panel-title">Edge supertypes</div>
      {supertype_options}
    </div>

    <div>
      <div class="panel-title">Action types</div>
      {action_options}
    </div>

    <div>
      <div class="panel-title">Edge supertype legend</div>
      {supertype_legend}
    </div>

    <div>
      <div class="panel-title">Action type legend</div>
      {action_legend}
    </div>

  </div>

  <div id="cy"></div>

  <div id="detail-panel">
    <span id="close-detail">✕</span>
    <div id="detail-title"></div>
    <div id="detail-body"></div>
    <div id="neighbor-list"></div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>

<script>
const ALL_ELEMENTS = {elements_json};

cytoscape.use(cytoscapeDagre);

const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: [],
  style: [
    {{
      selector: 'node',
      style: {{
        'background-color': 'data(color)',
        'label': 'data(label)',
        'font-size': '7px',
        'color': '#e0e0e0',
        'text-valign': 'bottom',
        'text-margin-y': '3px',
        'text-outline-width': 1,
        'text-outline-color': '#0f1117',
        'width': 18,
        'height': 18,
        'border-width': 0,
      }}
    }},
    {{
      selector: 'node:selected',
      style: {{
        'border-width': 3,
        'border-color': '#ffffff',
        'width': 24,
        'height': 24,
      }}
    }},
    {{
      selector: 'node.highlighted',
      style: {{ 'border-width': 2, 'border-color': '#ffdd44', 'width': 22, 'height': 22 }}
    }},
    {{
      selector: 'node.dimmed',
      style: {{ 'opacity': 0.15 }}
    }},
    {{
      selector: 'edge',
      style: {{
        'line-color': 'data(color)',
        'target-arrow-color': 'data(color)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'width': 1.5,
        'opacity': 0.7,
        'arrow-scale': 0.8,
      }}
    }},
    {{
      selector: 'edge:selected',
      style: {{ 'width': 3, 'opacity': 1 }}
    }},
    {{
      selector: 'edge.dimmed',
      style: {{ 'opacity': 0.05 }}
    }},
  ],
  layout: {{ name: 'preset' }},
  minZoom: 0.05,
  maxZoom: 5,
  wheelSensitivity: 0.2,
}});

// ---- State ----
let chLo = 1, chHi = 5, minConf = 0;
let activeSupertypes = new Set({json.dumps(supertypes)});
let activeActions = new Set({json.dumps(action_types)});

function getFilteredElements() {{
  const nodes = ALL_ELEMENTS.nodes.filter(n => {{
    const d = n.data;
    return d.chapter >= chLo && d.chapter <= chHi &&
           d.confidence >= minConf &&
           activeActions.has(d.actionType);
  }});
  const nodeIds = new Set(nodes.map(n => n.data.id));
  const edges = ALL_ELEMENTS.edges.filter(e => {{
    const d = e.data;
    return nodeIds.has(d.source) && nodeIds.has(d.target) &&
           activeSupertypes.has(d.supertype) &&
           d.confidence >= minConf;
  }});
  return {{ nodes, edges }};
}}

function applyLayout(name) {{
  const layoutConfig = name === 'dagre'
    ? {{ name: 'dagre', rankDir: 'LR', nodeSep: 30, edgeSep: 10, rankSep: 80, animate: false }}
    : name === 'cose'
    ? {{ name: 'cose', animate: false, randomize: true, nodeRepulsion: 4500, idealEdgeLength: 60 }}
    : {{ name, animate: false }};
  cy.layout(layoutConfig).run();
}}

function refresh() {{
  const {{ nodes, edges }} = getFilteredElements();
  cy.elements().remove();
  cy.add(nodes);
  cy.add(edges);
  updateStat();
  const layout = document.getElementById('layout-select').value;
  applyLayout(layout);
}}

function updateStat() {{
  document.getElementById('visible-stat').textContent =
    `${{cy.nodes().length}} events · ${{cy.edges().length}} links`;
}}

// ---- Controls ----
const chLoEl = document.getElementById('ch-lo');
const chHiEl = document.getElementById('ch-hi');
const chLoLabel = document.getElementById('ch-lo-label');
const chHiLabel = document.getElementById('ch-hi-label');
const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');

chLoEl.addEventListener('input', () => {{
  chLo = parseInt(chLoEl.value);
  if (chLo > chHi) {{ chHi = chLo; chHiEl.value = chHi; }}
  chLoLabel.textContent = chLo; chHiLabel.textContent = chHi;
  refresh();
}});
chHiEl.addEventListener('input', () => {{
  chHi = parseInt(chHiEl.value);
  if (chHi < chLo) {{ chLo = chHi; chLoEl.value = chLo; }}
  chLoLabel.textContent = chLo; chHiLabel.textContent = chHi;
  refresh();
}});
confSlider.addEventListener('input', () => {{
  minConf = parseFloat(confSlider.value);
  confVal.textContent = minConf.toFixed(2);
  refresh();
}});

document.querySelectorAll('.supertype-cb').forEach(cb => {{
  cb.addEventListener('change', () => {{
    if (cb.checked) activeSupertypes.add(cb.value);
    else activeSupertypes.delete(cb.value);
    refresh();
  }});
}});
document.querySelectorAll('.action-cb').forEach(cb => {{
  cb.addEventListener('change', () => {{
    if (cb.checked) activeActions.add(cb.value);
    else activeActions.delete(cb.value);
    refresh();
  }});
}});

document.getElementById('apply-layout').addEventListener('click', () => {{
  applyLayout(document.getElementById('layout-select').value);
}});
document.getElementById('fit-btn').addEventListener('click', () => cy.fit(undefined, 30));
document.getElementById('reset-btn').addEventListener('click', () => {{
  chLo = 1; chHi = 5; minConf = 0;
  chLoEl.value = 1; chHiEl.value = 5; confSlider.value = 0;
  chLoLabel.textContent = 1; chHiLabel.textContent = 5; confVal.textContent = '0.00';
  document.querySelectorAll('.supertype-cb, .action-cb').forEach(cb => cb.checked = true);
  activeSupertypes = new Set({json.dumps(supertypes)});
  activeActions = new Set({json.dumps(action_types)});
  refresh();
}});

// ---- Search ----
document.getElementById('search').addEventListener('input', function() {{
  const q = this.value.trim().toLowerCase();
  if (!q) {{ cy.elements().removeClass('dimmed highlighted'); return; }}
  const matched = cy.nodes().filter(n => n.data('fullName').toLowerCase().includes(q));
  cy.elements().addClass('dimmed');
  matched.removeClass('dimmed').addClass('highlighted');
  matched.connectedEdges().removeClass('dimmed');
}});

// ---- Detail panel ----
const detailPanel = document.getElementById('detail-panel');
const detailTitle = document.getElementById('detail-title');
const detailBody = document.getElementById('detail-body');
const neighborList = document.getElementById('neighbor-list');

function showNodeDetail(node) {{
  const d = node.data();
  detailTitle.textContent = d.fullName;
  detailBody.innerHTML = `
    <div class="detail-label">Action type</div>
    <div class="detail-row"><span>${{d.actionType}}</span></div>
    <div class="detail-label">Chapter</div>
    <div class="detail-row"><span>${{d.chapter}}</span></div>
    <div class="detail-label">Location</div>
    <div class="detail-row"><span>${{d.location || '—'}}</span></div>
    <div class="detail-label">Time</div>
    <div class="detail-row"><span>${{d.time || '—'}}</span></div>
    <div class="detail-label">Confidence</div>
    <div class="detail-row"><span>${{d.confidence.toFixed(2)}}</span></div>
  `;
  const outEdges = node.outgoers('edge');
  const inEdges = node.incomers('edge');
  let html = '';
  if (outEdges.length) {{
    html += '<div class="detail-label" style="margin-top:10px">Causes →</div>';
    outEdges.forEach(e => {{
      const tgt = e.target();
      html += `<div class="neighbor-item" data-id="${{tgt.id()}}">[<b>${{e.data('relationType')}}</b>] ${{tgt.data('label')}}</div>`;
    }});
  }}
  if (inEdges.length) {{
    html += '<div class="detail-label" style="margin-top:10px">← Caused by</div>';
    inEdges.forEach(e => {{
      const src = e.source();
      html += `<div class="neighbor-item" data-id="${{src.id()}}">[<b>${{e.data('relationType')}}</b>] ${{src.data('label')}}</div>`;
    }});
  }}
  neighborList.innerHTML = html;
  neighborList.querySelectorAll('.neighbor-item').forEach(el => {{
    el.addEventListener('click', () => {{
      const target = cy.getElementById(el.dataset.id);
      if (target.length) {{ cy.animate({{ center: {{ eles: target }}, zoom: 1.5 }}, {{ duration: 400 }}); target.select(); showNodeDetail(target); }}
    }});
  }});
  detailPanel.classList.add('visible');
}}

function showEdgeDetail(edge) {{
  const d = edge.data();
  detailTitle.textContent = d.relationType;
  detailBody.innerHTML = `
    <div class="detail-label">Supertype</div>
    <div class="detail-row"><span>${{d.supertype}}</span></div>
    <div class="detail-label">Mechanism</div>
    <div class="detail-row"><span>${{d.mechanism || '—'}}</span></div>
    <div class="detail-label">Confidence</div>
    <div class="detail-row"><span>${{d.confidence.toFixed(2)}}</span></div>
    <div class="detail-label">From</div>
    <div class="detail-row"><span>${{edge.source().data('label')}}</span></div>
    <div class="detail-label">To</div>
    <div class="detail-row"><span>${{edge.target().data('label')}}</span></div>
  `;
  neighborList.innerHTML = '';
  detailPanel.classList.add('visible');
}}

cy.on('tap', 'node', e => showNodeDetail(e.target));
cy.on('tap', 'edge', e => showEdgeDetail(e.target));
cy.on('tap', e => {{ if (e.target === cy) {{ detailPanel.classList.remove('visible'); cy.elements().removeClass('dimmed highlighted'); }} }});
document.getElementById('close-detail').addEventListener('click', () => detailPanel.classList.remove('visible'));

// ---- Init ----
chLoLabel.textContent = 1;
chHiLabel.textContent = 5;
refresh();
</script>
</body>
</html>
"""


def main():
    print("Loading events…")
    nodes = load_events()
    print(f"  {len(nodes)} events loaded")

    print("Loading causal links…")
    edges = load_edges()
    print(f"  {len(edges)} links loaded")

    print("Building HTML…")
    html = build_html(nodes, edges)

    out_path = os.path.join(PROJECT_ROOT, "graph_explorer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / 1_000_000
    print(f"Done → {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
