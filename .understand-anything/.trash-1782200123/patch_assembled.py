"""Patch assembled-graph.json: add missing import edge, fix types."""
import json

path = r'C:\Users\6seve\Codelib-severin\1_Research\Narrative-Causal-Graph\.understand-anything\intermediate\assembled-graph.json'
with open(path, encoding='utf-8') as f:
    g = json.load(f)

# Fix: debug.md should be document type, not file
for n in g['nodes']:
    if n['id'] == 'file:cekg_pipeline/debug.md':
        n['type'] = 'document'
        n['id'] = 'document:cekg_pipeline/debug.md'
        # Update edges referencing this node
        for e in g['edges']:
            if e['source'] == 'file:cekg_pipeline/debug.md':
                e['source'] = 'document:cekg_pipeline/debug.md'
            if e['target'] == 'file:cekg_pipeline/debug.md':
                e['target'] = 'document:cekg_pipeline/debug.md'
        print(f'Fixed debug.md type: file → document')
        break

# Add missing import: pipeline → dynamic_context
new_edge = {
    'source': 'file:cekg_pipeline/pipeline.py',
    'target': 'file:cekg_pipeline/dynamic_context.py',
    'type': 'imports',
    'direction': 'forward',
    'weight': 0.7
}

# Check if this edge already exists
exists = any(e['source'] == new_edge['source'] and e['target'] == new_edge['target'] for e in g['edges'])
if not exists:
    g['edges'].append(new_edge)
    print(f'Added missing import: pipeline.py → dynamic_context.py')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(g, f, ensure_ascii=False, indent=2)

print(f'Done. {len(g["nodes"])} nodes, {len(g["edges"])} edges')
