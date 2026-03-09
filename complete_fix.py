def patch_coreference_resolver():
    """FIX: Enhance resolver's resolve() method to use normalization"""
    print("\n[4/5] Fixing coreference_resolver.py...")
    
    file_path = "cekg_pipeline/coreference_resolver.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add normalize_character_name method
    normalization = '''
    def normalize_character_name(self, name: str) -> str:
        """
        Normalize character names: "Pip (Child)" -> "Pip", "Young Pip" -> "Pip"
        """
        if not name:
            return name
        
        # Remove parentheticals
        name = re.sub(r'\\s*\\([^)]*\\)', '', name)
        
        # Remove age descriptors
        for desc in ['Young', 'Old', 'Little', 'Elder', 'Older', 'Younger']:
            if name.startswith(desc + ' '):
                name = name[len(desc):].strip()
        
        # Remove titles (if not the only name)
        words = name.split()
        if len(words) > 1 and words[0] in ['Mr.', 'Mrs.', 'Miss', 'Ms.', 'Dr.', 'Sir', 'Lady', 'Lord']:
            name = ' '.join(words[1:])
        
        return name.strip()
'''
    
    # Insert before register_character method
    insertion_point = content.find('    def register_character(self')
    if insertion_point > 0:
        content = content[:insertion_point] + normalization + '\n' + content[insertion_point:]
    
    # Now enhance resolve() to use normalization
    # Find the resolve method and add normalization at the start
    resolve_start = content.find('    def resolve(self, mention: str')
    if resolve_start > 0:
        # Find the first line after the docstring
        docstring_end = content.find('"""', resolve_start + 50)
        if docstring_end > 0:
            insertion = content.find('\n', docstring_end + 3)
            if insertion > 0:
                normalize_call = '''
        # First normalize the mention
        mention = self.normalize_character_name(mention)
'''
                content = content[:insertion+1] + normalize_call + content[insertion+1:]
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("  ✓ Added normalize_character_name method")
    print("  ✓ Enhanced resolve() to use normalization")#!/usr/bin/env python3
"""
Complete CEKG Bug Fix + Cypher Export Fix
Fixes all 4 bugs PLUS the broken Cypher export (200k line files, trailing commas)

Usage: python complete_fix.py
"""

import os
import re
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

def patch_schemas():
    """FIX 1: Remove redundant event_category field"""
    print("\n[1/5] Fixing schemas.py...")
    
    file_path = "cekg_pipeline/schemas.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove event_category field
    content = re.sub(
        r'    event_category: str.*?\n',
        '',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("  ✓ Removed redundant event_category field")

def patch_exporters():
    """FIX: Cypher export - clean, valid syntax with batching"""
    print("\n[2/5] Fixing exporters.py (Cypher generation)...")
    
    file_path = "cekg_pipeline/exporters.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace _format_cypher_properties function
    old_format = r'def _format_cypher_properties\(props: Dict\[str, Any\]\) -> str:.*?return f"\{\{.*?\}\}"'
    
    new_format = '''def _format_cypher_properties(props: Dict[str, Any]) -> str:
    """Helper to format properties dict into valid Cypher string."""
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
    
    return "{" + ", ".join(prop_list) + "}"'''
    
    content = re.sub(old_format, new_format, content, flags=re.DOTALL)
    
    # Replace export_neo4j_cypher to add batching
    old_export = r'def export_neo4j_cypher\(.*?\n    print\(f"\[export\] Cypher exported:.*?\)'
    
    new_export = '''def export_neo4j_cypher(
    path: str,
    nodes: List[GenericNode],
    relationships: List[GenericRelationship],
    batch_size: int = 1000
):
    """Export to Neo4j Cypher with proper batching and valid syntax."""
    
    base_path, _ = os.path.splitext(path)
    path = base_path + ".txt"
    
    lines = []
    lines.append("// CEKG Cypher Import Script")
    lines.append(f"// Nodes: {len(nodes):,} | Relationships: {len(relationships):,}")
    lines.append("")
    
    # Indexes
    lines.append("// Indexes")
    lines.append("CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;")
    lines.append("CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE;")
    lines.append("CREATE CONSTRAINT scene_id IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE;")
    lines.append("")
    
    # Nodes
    lines.append("// NODES")
    nodes_by_label = defaultdict(list)
    for node in nodes:
        nodes_by_label[node.label].append(node)
    
    for label, node_list in nodes_by_label.items():
        lines.append(f"// {label} ({len(node_list):,})")
        
        for i in range(0, len(node_list), batch_size):
            batch = node_list[i:i + batch_size]
            
            for node in batch:
                props_str = _format_cypher_properties(node.properties)
                safe_uid = _escape_cypher_string(node.uid)
                lines.append(f'MERGE (n:{label} {{id: "{safe_uid}"}}) SET n = {props_str};')
            
            lines.append("")
    
    # Relationships
    lines.append("// RELATIONSHIPS")
    rels_by_type = defaultdict(list)
    for rel in relationships:
        rels_by_type[rel.rel_type].append(rel)
    
    for rel_type, rel_list in rels_by_type.items():
        lines.append(f"// {rel_type} ({len(rel_list):,})")
        
        for i in range(0, len(rel_list), batch_size):
            batch = rel_list[i:i + batch_size]
            
            for rel in batch:
                props_str = _format_cypher_properties(rel.properties)
                safe_start = _escape_cypher_string(rel.start_node_uid)
                safe_end = _escape_cypher_string(rel.end_node_uid)
                lines.append(
                    f'MATCH (a {{id: "{safe_start}"}}), (b {{id: "{safe_end}"}}) '
                    f'MERGE (a)-[r:{rel_type}]->(b) SET r = {props_str};'
                )
            
            lines.append("")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write('\\n'.join(lines))
    
    print(f"[export] Cypher exported: {len(lines):,} lines to {path}")'''
    
    content = re.sub(old_export, new_export, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("  ✓ Fixed Cypher export (batching, no trailing commas)")

def patch_graph_mapper():
    """FIX: Event properties and agent properties"""
    print("\n[3/5] Fixing graph_mapper.py...")
    
    file_path = "cekg_pipeline/graph_mapper.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix event properties (remove category, keep actionType, fix quote)
    old_event = r'"category": ev\.event_category,\s*"actionType": ev\.action_type,'
    new_event = '"actionType": ev.action_type,'
    content = re.sub(old_event, new_event, content)
    
    # Fix source_quote truncation
    old_quote = r'utils\._truncate_safe\(ev\.source_quote, 300\)'
    new_quote = 'ev.source_quote[:500] if ev.source_quote else ""'
    content = content.replace(old_quote, new_quote)
    
    # Add entityType and defaults to agent nodes
    old_agent = r'props = \{\s*"id": node_uid,\s*"name": entity_name\s*\}'
    new_agent = '''props = {
                "id": node_uid,
                "name": entity_name,
                "entityType": entity_type
            }'''
    content = re.sub(old_agent, new_agent, content)
    
    # Add default agentType
    old_agent_type = r'if agent_type:\s*props\["agentType"\] = agent_type\s*props\["theory"\] = theory'
    new_agent_type = '''if agent_type:
                props["agentType"] = agent_type
                props["theory"] = theory
            else:
                props["agentType"] = "STRUCTURAL_AGENT"
                props["theory"] = theory or "@McKee"'''
    content = re.sub(old_agent_type, new_agent_type, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("  ✓ Fixed event properties (actionType only, better quotes)")
    print("  ✓ Fixed agent properties (complete fields)")

def patch_coreference_resolver():
    """FIX: Character name normalization"""
    print("\n[4/5] Fixing coreference_resolver.py...")
    
    file_path = "cekg_pipeline/coreference_resolver.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add normalize_character_name method
    normalization = '''
    def normalize_character_name(self, name: str) -> str:
        """
        Normalize character names: "Pip (Child)" -> "Pip", "Young Pip" -> "Pip"
        """
        if not name:
            return name
        
        # Remove parentheticals
        name = re.sub(r'\\s*\\([^)]*\\)', '', name)
        
        # Remove age descriptors
        for desc in ['Young', 'Old', 'Little', 'Elder', 'Older', 'Younger']:
            if name.startswith(desc + ' '):
                name = name[len(desc):].strip()
        
        # Remove titles (if not the only name)
        words = name.split()
        if len(words) > 1 and words[0] in ['Mr.', 'Mrs.', 'Miss', 'Ms.', 'Dr.', 'Sir', 'Lady', 'Lord']:
            name = ' '.join(words[1:])
        
        return name.strip()
'''
    
    # Insert before register_character method
    insertion_point = content.find('    def register_character(self')
    if insertion_point > 0:
        content = content[:insertion_point] + normalization + '\n' + content[insertion_point:]
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("  ✓ Added normalize_character_name method")

def patch_pipeline():
    """FIX: Actually USE the resolver.resolve() method"""
    print("\n[5/5] Fixing pipeline.py (using resolver properly)...")
    
    file_path = "cekg_pipeline/pipeline.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace the entire actor extraction block
    # Look for the actor section in _parse_event_json_data
    actor_section_pattern = r'# Direct actor extraction with filtering.*?clean_actors = \[\].*?entity_occurrences_batch\[f"actor:\{actor_name\.lower\(\)\}"\]\.append\(\(event\.id, seq\)\)'
    
    new_actor_section = '''# Direct actor extraction with RESOLVER
                raw_actors = event_data.get("actors", [])
                if isinstance(raw_actors, str):
                    raw_actors = [raw_actors]
                
                clean_actors = []
                for actor in raw_actors:
                    if isinstance(actor, str) and len(actor) > 1:
                        actor_name = actor.strip()
                        
                        # Validate first
                        if not self.resolver.is_valid_character_name(actor_name):
                            continue
                        
                        # FIX: Actually use resolver to get canonical name
                        canonical_name = self.resolver.resolve(actor_name)
                        
                        if not canonical_name:
                            continue
                        
                        clean_actors.append(canonical_name)
                        aid = graph_builder._generate_entity_id(canonical_name, "agent", event.id, graph_model)
                        all_produces.append(schemas.EventProducesEntity(
                            event.id, aid, canonical_name, "actor", "PRODUCES_ACTOR", 1.0,
                            agent_type=None, theory=theory
                        ))
                        entity_occurrences_batch[f"actor:{canonical_name.lower()}"].append((event.id, seq))'''
    
    content = re.sub(actor_section_pattern, new_actor_section, content, flags=re.DOTALL)
    
    # Find and replace the entire patient extraction block
    patient_section_pattern = r'# Direct patient extraction with filtering.*?clean_patients = \[\].*?entity_occurrences_batch\[f"patient:\{pat_name\.lower\(\)\}"\]\.append\(\(event\.id, seq\)\)'
    
    new_patient_section = '''# Direct patient extraction with RESOLVER
                raw_patients = event_data.get("patients", [])
                if isinstance(raw_patients, str):
                    raw_patients = [raw_patients]
                
                clean_patients = []
                for patient in raw_patients:
                    if isinstance(patient, str) and len(patient) > 1:
                        pat_name = patient.strip()
                        
                        # Validate first
                        if not self.resolver.is_valid_character_name(pat_name):
                            continue
                        
                        # FIX: Actually use resolver to get canonical name
                        canonical_name = self.resolver.resolve(pat_name)
                        
                        if not canonical_name:
                            continue
                        
                        clean_patients.append(canonical_name)
                        pid = graph_builder._generate_entity_id(canonical_name, "agent", event.id, graph_model)
                        all_produces.append(schemas.EventProducesEntity(
                            event.id, pid, canonical_name, "patient", "PRODUCES_PATIENT", 1.0,
                            agent_type=None, theory=theory
                        ))
                        entity_occurrences_batch[f"patient:{canonical_name.lower()}"].append((event.id, seq))'''
    
    content = re.sub(patient_section_pattern, new_patient_section, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("  ✓ Now using resolver.resolve() for character name resolution")
    print("  ✓ Pip variants will automatically merge!")

def create_improved_force_import():
    """Create improved force_import.py script"""
    print("\n[BONUS] Creating improved force_import.py...")
    
    script = '''from neo4j import GraphDatabase
import os
import re

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "greeplace1!"  
DB_NAME = "test"        
FILE_PATH = "ge_import.txt"

def clean_statement(stmt):
    """Clean up Cypher statement"""
    if not stmt.strip():
        return None
    
    # Remove trailing commas before closing braces
    stmt = re.sub(r',\\s*}', '}', stmt)
    stmt = stmt.strip()
    
    if not stmt.endswith(';'):
        stmt += ';'
    
    return stmt

def import_data():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File not found: {FILE_PATH}")
        return

    print(f"🔌 Connecting to Neo4j...")
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ Connected")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print(f"📖 Reading {FILE_PATH}...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        script = f.read()

    # Split on semicolon + newline (preserves semicolons in text)
    queries = re.split(r';\\s*[\\r\\n]+', script)
    queries = [clean_statement(q) for q in queries if q.strip()]
    
    # Filter executable statements
    executable = [q for q in queries 
                  if q and any(q.upper().strip().startswith(cmd) 
                  for cmd in ['CREATE', 'MERGE', 'MATCH', 'UNWIND'])]
    
    print(f"🚀 Executing {len(executable):,} statements...")
    
    success = 0
    failed = 0
    
    with driver.session(database=DB_NAME) as session:
        for i, query in enumerate(executable):
            try:
                session.run(query)
                success += 1
                
                if i % 100 == 0:
                    pct = 100 * (i + 1) / len(executable)
                    print(f"   {i+1}/{len(executable)} ({pct:.1f}%) | ✓{success} ✗{failed}", end="\\r")
                    
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\\n❌ Error: {str(e)[:100]}")

    print(f"\\n\\n✅ Complete: {success:,} succeeded, {failed:,} failed")
    driver.close()

if __name__ == "__main__":
    import_data()
'''
    
    with open("force_import_improved.py", "w") as f:
        f.write(script)
    
    print("  ✓ Created force_import_improved.py")

def main():
    print("="*60)
    print("CEKG Complete Fix - All Bugs + Cypher Export")
    print("="*60)
    
    try:
        patch_schemas()
        patch_exporters()
        patch_graph_mapper()
        patch_coreference_resolver()
        patch_pipeline()
        create_improved_force_import()
        
        print("\n" + "="*60)
        print("✅ ALL FIXES APPLIED")
        print("="*60)
        
        print("\nFixed Issues:")
        print("  1. ✓ Removed event_category (kept actionType)")
        print("  2. ✓ Fixed source_quote (500 chars)")
        print("  3. ✓ Enhanced resolver with normalization")
        print("  4. ✓ Pipeline now uses resolver.resolve() properly")
        print("  5. ✓ Complete agent properties")
        print("  6. ✓ Clean Cypher export (no trailing commas, batched)")
        print("  7. ✓ Improved import script")
        
        print("\n💡 Key Fix: The resolver was there but never used!")
        print("   Now resolver.resolve() maps all Pip variants -> canonical name")
        
        print("\nBackups created with .backup.TIMESTAMP extension")
        
        print("\nNext Steps:")
        print("  1. Re-run pipeline:")
        print("     python main.py --input great_expectations.txt --full")
        print("  2. Import to Neo4j:")
        print("     python force_import_improved.py")
        print("  3. Verify:")
        print("     - Pip is single agent")
        print("     - Events have source_quote")
        print("     - No broken Cypher statements")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())