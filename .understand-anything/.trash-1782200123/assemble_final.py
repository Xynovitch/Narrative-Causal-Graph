"""Assemble final knowledge-graph.json from all intermediate files."""
import json, datetime

base = r'C:\Users\6seve\Codelib-severin\1_Research\Narrative-Causal-Graph\.understand-anything'

# Read assembled graph
with open(f'{base}/intermediate/assembled-graph.json', encoding='utf-8') as f:
    ag = json.load(f)

# Read layers
with open(f'{base}/intermediate/layers.json', encoding='utf-8') as f:
    layers_data = json.load(f)
layers = layers_data.get('layers', layers_data if isinstance(layers_data, list) else [])

# Read tour
with open(f'{base}/intermediate/tour.json', encoding='utf-8') as f:
    tour_data = json.load(f)
tour = tour_data.get('steps', tour_data if isinstance(tour_data, list) else [])

# Read scan result for project info
with open(f'{base}/intermediate/scan-result.json', encoding='utf-8') as f:
    scan = json.load(f)

# Normalize layers
norm_layers = []
for l in layers:
    nl = {
        'id': l.get('id', f'layer:{l["name"]}'),
        'name': l.get('name', ''),
        'description': l.get('description', ''),
        'nodeIds': l.get('nodeIds', l.get('nodes', []))
    }
    # Convert bare paths to file: prefixed
    fixed_ids = []
    for nid in nl['nodeIds']:
        if nid and not any(nid.startswith(p) for p in ['file:', 'config:', 'document:', 'function:', 'class:', 'service:', 'pipeline:']):
            nid = 'file:' + nid
        fixed_ids.append(nid)
    nl['nodeIds'] = fixed_ids
    norm_layers.append(nl)

# Normalize tour
norm_tour = []
for i, s in enumerate(tour):
    ns = {
        'order': s.get('order', i+1),
        'title': s.get('title', ''),
        'description': s.get('description', ''),
        'nodeIds': s.get('nodeIds', s.get('nodesToInspect', []))
    }
    # Fix class: and function: IDs that might be missing cekg_pipeline/ prefix
    fixed_ids = []
    for nid in ns['nodeIds']:
        if nid and nid.startswith('class:') and '/cekg_pipeline/' not in nid and not nid.startswith('class:cekg_pipeline/'):
            # Check if it's just class:filename:Classname - needs cekg_pipeline/ prefix
            parts = nid.split(':')
            if len(parts) == 3 and not parts[1].startswith('cekg_pipeline/'):
                new_nid = f'class:cekg_pipeline/{parts[1]}:{parts[2]}'
                fixed_ids.append(new_nid)
                continue
        if nid and nid.startswith('function:') and '/cekg_pipeline/' not in nid and not nid.startswith('function:cekg_pipeline/') and not nid.startswith('function:main.py'):
            parts = nid.split(':')
            if len(parts) >= 3 and not parts[1].startswith('cekg_pipeline/'):
                new_nid = f'function:cekg_pipeline/{parts[1]}:{parts[2]}'
                fixed_ids.append(new_nid)
                continue
        fixed_ids.append(nid)
    ns['nodeIds'] = fixed_ids
    norm_tour.append(ns)

# Build final graph
graph = {
    'version': '1.0.0',
    'project': {
        'name': scan.get('projectName', 'Narrative-Causal-Graph'),
        'languages': scan.get('languages', ['python']),
        'frameworks': scan.get('frameworks', []),
        'description': scan.get('projectDescription', ''),
        'analyzedAt': datetime.datetime.now().isoformat(),
        'gitCommitHash': 'f602e040e670aece183dcf24fbc6999d8690de8e'
    },
    'nodes': ag.get('nodes', []),
    'edges': ag.get('edges', []),
    'layers': norm_layers,
    'tour': norm_tour
}

with open(f'{base}/intermediate/assembled-graph.json', 'w', encoding='utf-8') as f:
    json.dump(graph, f, ensure_ascii=False, indent=2)

print(f'Final graph: {len(graph["nodes"])} nodes, {len(graph["edges"])} edges, {len(norm_layers)} layers, {len(norm_tour)} tour steps')
