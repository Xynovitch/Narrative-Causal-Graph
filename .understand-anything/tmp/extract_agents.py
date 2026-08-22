import re
t = open(r'C:\Users\6seve\Codelib-severin\1_Research\Narrative-Causal-Graph\schema.json', encoding='utf-8').read()
# Extract all agentType values from neo4jProperties
names = re.findall(r'"agentType": "([A-Z][A-Z_]+)"', t)
explanations = re.findall(r'"name": "([A-Z][A-Z_]+)"\s*,\s*"explanation": "([^"]+)"', t)

# Build lookup for explanations that match agent types
exp_lookup = {}
for n, e in explanations:
    exp_lookup[n] = e

print(f'Agent Types: {len(names)} total\n')
for n in names:
    exp = exp_lookup.get(n, '')
    print(f'  {n}: {exp}')
