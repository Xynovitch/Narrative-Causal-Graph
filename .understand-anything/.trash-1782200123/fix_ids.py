"""Fix class: node IDs to include cekg_pipeline/ prefix, fix tour references."""
import json

path = r'C:\Users\6seve\Codelib-severin\1_Research\Narrative-Causal-Graph\.understand-anything\intermediate\assembled-graph.json'
with open(path, encoding='utf-8') as f:
    g = json.load(f)

# Map old class IDs to new ones (add cekg_pipeline/ prefix)
id_map = {}
for n in g['nodes']:
    if n['type'] == 'class' and not n['id'].startswith('class:cekg_pipeline/'):
        parts = n['id'].split(':', 1)[1]  # Remove "class:"
        new_id = f'class:cekg_pipeline/{parts}'
        id_map[n['id']] = new_id
        n['id'] = new_id
        print(f'Fixed class: {n["id"]}')

# Update edge references
for e in g['edges']:
    if e['source'] in id_map:
        e['source'] = id_map[e['source']]
    if e['target'] in id_map:
        e['target'] = id_map[e['target']]

# Update tour references (fix nodeIds that reference class: with bare paths)
for step in g['tour']:
    new_ids = []
    for nid in step['nodeIds']:
        if nid.startswith('class:') and not nid.startswith('class:cekg_pipeline/') and nid != 'class:pipeline.py:CEKGPreprocessor':
            # Already has the correct format? Check if it's class:cekg_pipeline/ format
            if '/cekg_pipeline/' in nid:
                new_ids.append(nid)
            else:
                # Try to fix: class:xxx.py:Name → class:cekg_pipeline/xxx.py:Name
                after_class = nid[6:]  # Remove "class:"
                fixed = f'class:cekg_pipeline/{after_class}'
                new_ids.append(fixed)
        elif nid == 'class:pipeline.py:CEKGPreprocessor':
            new_ids.append('class:cekg_pipeline/pipeline.py:CEKGPreprocessor')
        elif nid.startswith('function:') and nid.startswith('function:cekg_pipeline/') and nid.count('/') == 1:
            # Format is function:cekg_pipeline/filename.py:name() — should be function:cekg_pipeline/path/filename.py:name()
            # This should be fine, just keep it
            new_ids.append(nid)
        else:
            new_ids.append(nid)
    step['nodeIds'] = new_ids

# Update layer references too
for l in g['layers']:
    new_ids = []
    for nid in l['nodeIds']:
        if nid in id_map:
            new_ids.append(id_map[nid])
        else:
            new_ids.append(nid)
    l['nodeIds'] = new_ids

with open(path, 'w', encoding='utf-8') as f:
    json.dump(g, f, ensure_ascii=False, indent=2)

print(f'Done. {len(id_map)} class IDs fixed.')
