"""Fix batch-3 JSON format: uid→id, label→name, from→source, to→target, etc."""
import json, sys

intermediate = r'C:\Users\6seve\Codelib-severin\1_Research\Narrative-Causal-Graph\.understand-anything\intermediate'

for fname in ['batch-3-part-1.json', 'batch-3-part-2.json']:
    path = intermediate + '\\' + fname
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    
    nodes = d.get('nodes', [])
    edges = d.get('edges', [])
    
    fixed_nodes = []
    for n in nodes:
        fn = {}
        fn['id'] = n.get('uid', n.get('id', ''))
        fn['name'] = n.get('label', n.get('name', ''))
        fn['type'] = (n.get('type', 'file') or 'file').lower()
        fn['filePath'] = n.get('path', '')
        fn['summary'] = n.get('summary', n.get('description', ''))
        fn['tags'] = n.get('tags', [])
        if not fn['tags']:
            fn['tags'] = ['python-code'] if fn['filePath'].endswith('.py') else ['documentation']
        fn['complexity'] = 'moderate'  # default
        totalLines = n.get('totalLines', 0)
        if totalLines < 50:
            fn['complexity'] = 'simple'
        elif totalLines > 300:
            fn['complexity'] = 'complex'
        # lineRange
        if 'lineRange' in n:
            fn['lineRange'] = n['lineRange']
        fixed_nodes.append(fn)
    
    fixed_edges = []
    for e in edges:
        fe = {}
        fe['source'] = e.get('from', e.get('source', ''))
        fe['target'] = e.get('to', e.get('target', ''))
        fe['type'] = e.get('type', 'imports')
        fe['direction'] = e.get('direction', 'forward')
        fe['weight'] = e.get('weight', 0.7)
        if fe['type'] == 'contains':
            fe['weight'] = 1.0
        elif fe['type'] == 'calls':
            fe['weight'] = 0.8
        fixed_edges.append(fe)
    
    out = {'nodes': fixed_nodes, 'edges': fixed_edges}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    print(f'Fixed {fname}: {len(fixed_nodes)} nodes, {len(fixed_edges)} edges')

# Also fix batch-2 (novel files need "file:" prefix)
path2 = intermediate + '\\batch-2.json'
with open(path2, encoding='utf-8') as f:
    d2 = json.load(f)

fixed = []
for n in d2.get('nodes', []):
    nid = n.get('id', '')
    if nid and not nid.startswith('file:') and not nid.startswith('config:') and not nid.startswith('document:'):
        n['id'] = 'file:' + nid
    if not n.get('name'):
        n['name'] = n.get('id', '').split('/')[-1] if '/' in n.get('id', '') else n.get('id', '')
    if not n.get('filePath'):
        n['filePath'] = n.get('id', '').replace('file:', '')
    if not n.get('tags'):
        n['tags'] = ['小说数据', '输入文本', '文学']
    if not n.get('summary'):
        n['summary'] = '小说文本输入数据'
    if not n.get('complexity'):
        n['complexity'] = 'moderate'
    fixed.append(n)

out2 = {'nodes': fixed, 'edges': d2.get('edges', [])}
with open(path2, 'w', encoding='utf-8') as f:
    json.dump(out2, f, ensure_ascii=False, indent=2)
print(f'Fixed batch-2.json: {len(fixed)} nodes')
