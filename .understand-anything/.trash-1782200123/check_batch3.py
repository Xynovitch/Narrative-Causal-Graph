import json, sys
intermediate = r'C:\Users\6seve\Codelib-severin\1_Research\Narrative-Causal-Graph\.understand-anything\intermediate'

for fname in ['batch-3-part-1.json', 'batch-3-part-2.json', 'batch-1.json', 'batch-2.json']:
    path = intermediate + '\\' + fname
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    nodes = d.get('nodes', [])
    edges = d.get('edges', [])
    print(f'=== {fname}: {len(nodes)} nodes, {len(edges)} edges ===')
    for i, n in enumerate(nodes[:3]):
        print(f'  [{i}] id={n.get("id","MISSING")}  type={n.get("type","?")}  name={n.get("name","?")}')
    if len(nodes) > 6:
        print(f'  ... ({len(nodes)-6} more) ...')
    for i, n in enumerate(nodes[-3:], len(nodes)-3):
        print(f'  [{i}] id={n.get("id","MISSING")}  type={n.get("type","?")}  name={n.get("name","?")}')
    # Check edges
    bad_edges = [e for e in edges if not e.get('source') or not e.get('target')]
    if bad_edges:
        print(f'  WARNING: {len(bad_edges)} edges with empty source/target!')
    print()
