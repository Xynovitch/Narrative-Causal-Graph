import pandas as pd
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
node_files = [
    {'file': 'agents.csv',      'type': 'Agent',     'label_col': 'name'},
    {'file': 'scenes.csv',      'type': 'Scene',     'label_col': 'theme'},
    {'file': 'events.csv',      'type': 'Event',     'label_col': 'name'},
    {'file': 'whyfactors.csv',  'type': 'WhyFactor', 'label_col': 'factor'},
]

edge_files = [
    'acts_in.csv',
    'affected_in.csv',
    'causal_links.csv',
    'causes.csv',
    'follows.csv',
    'motivates.csv',
    'scene_includes_event.csv',
    'semantic_links.csv'
]

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def find_col(df, candidates):
    """Finds the first matching column from a list of candidates."""
    for col in df.columns:
        if any(c in col.strip() for c in candidates): return col
    return None

# -----------------------------------------------------------------------------
# 1. PRE-CALCULATE SCENE CONTEXT (Optional Metadata)
# -----------------------------------------------------------------------------
# We map events to scenes just in case you want to inspect them, 
# but we WON'T use this for the main color (to avoid the "monotone" look).
event_to_scene_map = {}
if os.path.exists('scenes.csv') and os.path.exists('scene_includes_event.csv'):
    print("Mapping Events to Scenes for metadata...")
    df_s = pd.read_csv('scenes.csv')
    df_l = pd.read_csv('scene_includes_event.csv')
    
    # Create ID -> Name dictionary
    sid = find_col(df_s, ['ID', 'id'])
    stheme = find_col(df_s, ['theme', 'name'])
    s_dict = dict(zip(df_s[sid], df_s[stheme]))
    
    # Map Links
    l_src = find_col(df_l, ['START', 'source']) # Scene
    l_tgt = find_col(df_l, ['END', 'target'])   # Event
    
    for _, row in df_l.iterrows():
        # Get scene name
        s_name = s_dict.get(row[l_src], "Unknown Scene") 
        event_to_scene_map[row[l_tgt]] = s_name

# -----------------------------------------------------------------------------
# 2. PROCESS NODES
# -----------------------------------------------------------------------------
all_nodes = []
print("--- Processing Nodes ---")

for item in node_files:
    filename = item['file']
    if not os.path.exists(filename):
        print(f"Skipping {filename}")
        continue
    
    print(f"Reading {filename}...")
    df = pd.read_csv(filename)
    
    id_col = find_col(df, [':ID', 'ID', 'id'])
    label_col = find_col(df, [item.get('label_col'), 'label', 'name', 'factor'])
    
    clean_df = pd.DataFrame()
    clean_df['id'] = df[id_col]
    clean_df['label'] = df[label_col] if label_col else df[id_col]
    clean_df['type'] = item['type']
    
    # --- COLOR STRATEGY ---
    # For Events: Use 'event_category' to keep the Dynamic Color Range.
    # For Others: Use their Type (Agent, Scene, WhyFactor).
    if item['type'] == 'Event':
        cat_col = find_col(df, ['event_category', 'category'])
        clean_df['group'] = df[cat_col].fillna('Uncategorized Event')
        # Add Scene Name as an extra column (doesn't affect color, but visible on click)
        clean_df['scene_name'] = clean_df['id'].map(event_to_scene_map).fillna('No Scene')
        # Events are small
        clean_df['size'] = 1
    else:
        # Agents/Scenes/WhyFactors get their own distinct color
        clean_df['group'] = item['type'] 
        clean_df['scene_name'] = 'N/A'
        # Make these Nodes BIG so they act as "Hubs"
        clean_df['size'] = 15 

    all_nodes.append(clean_df)

final_nodes = pd.concat(all_nodes, ignore_index=True)
final_nodes.drop_duplicates(subset=['id'], inplace=True)
final_nodes.to_csv('cosmograph_nodes.csv', index=False)
print(f"SUCCESS: Exported {len(final_nodes)} nodes.")

# -----------------------------------------------------------------------------
# 3. PROCESS EDGES
# -----------------------------------------------------------------------------
all_edges = []
print("\n--- Processing Edges ---")

for filename in edge_files:
    if not os.path.exists(filename): continue
    
    df = pd.read_csv(filename)
    src_col = find_col(df, [':START_ID', 'source', 'start'])
    tgt_col = find_col(df, [':END_ID', 'target', 'end'])
    
    if src_col and tgt_col:
        clean_df = pd.DataFrame()
        clean_df['source'] = df[src_col]
        clean_df['target'] = df[tgt_col]
        all_edges.append(clean_df)

final_edges = pd.concat(all_edges, ignore_index=True)
final_edges.to_csv('cosmograph_edges.csv', index=False)
print(f"SUCCESS: Exported {len(final_edges)} edges.")