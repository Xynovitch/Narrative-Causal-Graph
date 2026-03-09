#!/usr/bin/env python3
"""
Emergency fix for exporters.py syntax error
Run this to fix the broken exporters.py file
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create backup of file before modifying"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"  ✓ Backed up to {backup_path}")
        return backup_path
    return None

def fix_exporters():
    """Fix the broken exporters.py file"""
    print("\n[EMERGENCY FIX] Fixing exporters.py syntax error...")
    
    file_path = "cekg_pipeline/exporters.py"
    
    # Restore from backup if it exists
    backups = [f for f in os.listdir("cekg_pipeline") if f.startswith("exporters.py.backup")]
    if backups:
        latest_backup = sorted(backups)[-1]
        print(f"  Restoring from {latest_backup}...")
        shutil.copy2(f"cekg_pipeline/{latest_backup}", file_path)
        print("  ✓ Restored original file")
    
    # Now apply the correct fix
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the _format_cypher_properties function
    # Look for the function definition
    start_marker = "def _format_cypher_properties(props: Dict[str, Any]) -> str:"
    end_marker = "def export_neo4j_cypher("
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("  ✗ Could not find function markers")
        return False
    
    # Replace the function
    new_function = '''def _format_cypher_properties(props: Dict[str, Any]) -> str:
    """Helper to format a properties dictionary into a Cypher string."""
    if not props:
        return "{}"
    
    prop_list = []
    for k, v in props.items():
        if v is None:
            continue
        
        safe_key = str(k).replace('"', '\\\\"')
        
        if isinstance(v, str):
            safe_val = _escape_cypher_string(v)
            prop_list.append(f'{safe_key}: "{safe_val}"')
        elif isinstance(v, bool):
            prop_list.append(f'{safe_key}: {str(v).lower()}')
        elif isinstance(v, (int, float)):
            prop_list.append(f'{safe_key}: {v}')
        else:
            safe_val = _escape_cypher_string(str(v))
            prop_list.append(f'{safe_key}: "{safe_val}"')
    
    return "{" + ", ".join(prop_list) + "}"


'''
    
    content = content[:start_idx] + new_function + content[end_idx:]
    
    # Now fix the export_neo4j_cypher function
    start_marker = "def export_neo4j_cypher("
    # Find the end of this function (next def or end of file)
    start_idx = content.find(start_marker)
    
    if start_idx == -1:
        print("  ✗ Could not find export_neo4j_cypher")
        return False
    
    # Find next function or class definition
    search_start = start_idx + len(start_marker)
    next_def = content.find("\ndef ", search_start)
    next_class = content.find("\nclass ", search_start)
    
    end_idx = min(x for x in [next_def, next_class, len(content)] if x > 0)
    
    new_export_function = '''def export_neo4j_cypher(
    path: str,
    nodes: List[GenericNode],
    relationships: List[GenericRelationship],
    batch_size: int = 1000
):
    """
    Export a generic graph to a Neo4j Cypher script.
    Fixed: Generates clean, valid Cypher with proper batching.
    """
    
    base_path, _ = os.path.splitext(path)
    path = base_path + ".txt"
    
    lines = []
    lines.append("// ============================================================")
    lines.append("// CEKG Cypher Import Script (Generated)")
    lines.append(f"// Total Nodes: {len(nodes)}")
    lines.append(f"// Total Relationships: {len(relationships)}")
    lines.append("// ============================================================")
    lines.append("")
    
    # 1. Create Constraint/Index setup
    lines.append("// --- INDEXES ---")
    lines.append("CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;")
    lines.append("CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE;")
    lines.append("CREATE CONSTRAINT scene_id IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE;")
    lines.append("CREATE INDEX event_sequence IF NOT EXISTS FOR (e:Event) ON (e.sequence);")
    lines.append("CREATE INDEX event_chapter IF NOT EXISTS FOR (e:Event) ON (e.chapter);")
    lines.append("")
    
    # 2. Create Nodes in Batches
    lines.append("// ============================================================")
    lines.append("// NODES (Batched)")
    lines.append("// ============================================================")
    lines.append("")
    
    nodes_by_label = defaultdict(list)
    for node in nodes:
        nodes_by_label[node.label].append(node)
    
    statement_count = 0
    
    for label, node_list in nodes_by_label.items():
        lines.append(f"// --- {label} Nodes ({len(node_list):,}) ---")
        
        for i in range(0, len(node_list), batch_size):
            batch = node_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(node_list) + batch_size - 1) // batch_size
            
            if len(node_list) > batch_size:
                lines.append(f"// Batch {batch_num}/{total_batches}")
            
            for node in batch:
                props_str = _format_cypher_properties(node.properties)
                safe_uid = _escape_cypher_string(node.uid)
                
                lines.append(f'MERGE (n:{label} {{id: "{safe_uid}"}}) SET n = {props_str};')
                statement_count += 1
            
            lines.append("")
    
    lines.append("// ============================================================")
    lines.append("// RELATIONSHIPS (Batched)")
    lines.append("// ============================================================")
    lines.append("")
    
    # 3. Create Relationships in Batches
    rels_by_type = defaultdict(list)
    for rel in relationships:
        rels_by_type[rel.rel_type].append(rel)
    
    for rel_type, rel_list in rels_by_type.items():
        lines.append(f"// --- {rel_type} Relationships ({len(rel_list):,}) ---")
        
        for i in range(0, len(rel_list), batch_size):
            batch = rel_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(rel_list) + batch_size - 1) // batch_size
            
            if len(rel_list) > batch_size:
                lines.append(f"// Batch {batch_num}/{total_batches}")
            
            for rel in batch:
                props_str = _format_cypher_properties(rel.properties)
                safe_start_uid = _escape_cypher_string(rel.start_node_uid)
                safe_end_uid = _escape_cypher_string(rel.end_node_uid)
                
                lines.append(
                    f'MATCH (a {{id: "{safe_start_uid}"}}), (b {{id: "{safe_end_uid}"}}) '
                    f'MERGE (a)-[r:{rel_type}]->(b) SET r = {props_str};'
                )
                statement_count += 1
            
            lines.append("")
    
    lines.append("// ============================================================")
    lines.append("// IMPORT COMPLETE")
    lines.append(f"// Total Statements: {statement_count:,}")
    lines.append("// ============================================================")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write('\\n'.join(lines))
    
    print(f"[export] Cypher exported: {statement_count:,} statements to {path}")
    print(f"[export] File size: {len(lines):,} lines")


'''
    
    content = content[:start_idx] + new_export_function + content[end_idx:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  ✓ Fixed exporters.py")
    return True

def main():
    print("="*60)
    print("Emergency Fix for exporters.py")
    print("="*60)
    
    try:
        if fix_exporters():
            print("\n✅ exporters.py fixed successfully!")
            print("\nYou can now run:")
            print("  python main.py --input great_expectations.txt --full")
        else:
            print("\n✗ Fix failed - please check the file manually")
            return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())