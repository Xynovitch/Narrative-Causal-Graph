"""Fix batch-2 edges to use file: prefix."""
import json

path = r'C:\Users\6seve\Codelib-severin\1_Research\Narrative-Causal-Graph\.understand-anything\intermediate\batch-2.json'
with open(path, encoding='utf-8') as f:
    d = json.load(f)

fixed_edges = []
for e in d.get('edges', []):
    src = e.get('source', '')
    tgt = e.get('target', '')
    if src and not src.startswith('file:') and not src.startswith('config:') and not src.startswith('document:') and not src.startswith('function:') and not src.startswith('class:'):
        src = 'file:' + src
    if tgt and not tgt.startswith('file:') and not tgt.startswith('config:') and not tgt.startswith('document:') and not tgt.startswith('function:') and not tgt.startswith('class:'):
        tgt = 'file:' + tgt
    e['source'] = src
    e['target'] = tgt
    if not e.get('direction'):
        e['direction'] = 'forward'
    if not e.get('weight'):
        e['weight'] = 0.5
    fixed_edges.append(e)

out = {'nodes': d['nodes'], 'edges': fixed_edges}
with open(path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'Fixed batch-2.json: {len(fixed_edges)} edges')
