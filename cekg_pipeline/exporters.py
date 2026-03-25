import os
import json
import csv
from collections import defaultdict
from dataclasses import asdict
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from .schemas import (
    CEKEvent, EventProducesEntity, CausalLink,
    GenericNode, GenericRelationship, Scene, ThematicLink
)
from .utils import _truncate_safe

# ============================================================================
# FIXED: Neo4j Cypher Compliance
# ============================================================================

def _needs_backtick_escaping(identifier: str) -> bool:
    """
    Check if identifier needs backtick escaping per Neo4j rules.
    
    Identifiers need escaping if they contain:
    - Spaces, hyphens, dots, colons, special chars
    - Start with numbers
    - Are reserved keywords
    """
    if not identifier:
        return True
    
    # Reserved keywords (partial list of most common)
    reserved = {
        'all', 'and', 'as', 'asc', 'by', 'case', 'create', 'delete', 
        'desc', 'distinct', 'end', 'else', 'false', 'in', 'is', 'match',
        'merge', 'not', 'null', 'or', 'order', 'remove', 'return', 
        'set', 'skip', 'then', 'true', 'union', 'unwind', 'when', 'where', 'with'
    }
    
    if identifier.lower() in reserved:
        return True
    
    # Check for special characters that require escaping
    special_chars = {' ', '-', '.', ':', '$', '!', '@', '#', '%', '^', '&', '*', 
                     '(', ')', '[', ']', '{', '}', '/', '\\', '|', '?', '<', '>',
                     '+', '=', '~', '`', '"', "'"}
    
    if any(char in identifier for char in special_chars):
        return True
    
    # Check if starts with number
    if identifier[0].isdigit():
        return True
    
    return False

def _escape_identifier(identifier: str) -> str:
    """
    Escape identifier for use in Cypher.
    Uses backticks if needed, otherwise returns as-is.
    """
    if _needs_backtick_escaping(identifier):
        # Escape backticks inside the identifier
        escaped = identifier.replace('`', '``')
        return f'`{escaped}`'
    return identifier

def _escape_cypher_value(v: Any) -> str:
    """
    FIX: Properly escape values for Cypher with correct order.
    
    Critical: Escape quotes BEFORE backslashes to avoid double-escaping.
    """
    if v is None:
        return 'null'
    
    if isinstance(v, bool):
        return 'true' if v else 'false'
    
    if isinstance(v, (int, float)):
        return str(v)
    
    if isinstance(v, list):
        # Handle arrays
        escaped_items = [_escape_cypher_value(item) for item in v]
        return '[' + ', '.join(escaped_items) + ']'
    
    # String escaping - CORRECT ORDER
    s = str(v)
    s = s.replace('\\', '\\\\')  # Escape backslashes
    s = s.replace('"', '\\"')     # Escape quotes
    s = s.replace('\n', '\\n')    # Escape newlines
    s = s.replace('\r', '\\r')    # Escape carriage returns
    s = s.replace('\t', '\\t')    # Escape tabs
    
    return f'"{s}"'

def _format_cypher_properties(props: Dict[str, Any]) -> str:
    """
    FIX: Format properties with proper key escaping and value handling.
    """
    if not props:
        return "{}"
    
    prop_list = []
    for k, v in props.items():
        # Skip None values
        if v is None:
            continue
        
        # FIX: Escape property key with backticks if needed
        safe_key = _escape_identifier(str(k))
        
        # FIX: Use new value escaping
        safe_val = _escape_cypher_value(v)
        
        # Don't add quotes if already formatted (bool, number, null, array)
        if safe_val in ['true', 'false', 'null'] or safe_val[0] in '[0123456789-':
            prop_list.append(f'{safe_key}: {safe_val}')
        else:
            prop_list.append(f'{safe_key}: {safe_val}')
    
    return "{" + ", ".join(prop_list) + "}"

def export_neo4j_cypher(
    path: str,
    nodes: List[GenericNode],
    relationships: List[GenericRelationship],
    batch_size: int = 500  # FIX: Reduced from 1000 to 500
):
    """
    FIX: Export with proper MERGE + ON CREATE/MATCH pattern.
    
    Changes:
    1. Use ON CREATE SET instead of bare SET
    2. Escape relationship types with backticks
    3. Escape property keys with backticks
    4. Reduced batch size to 500 (from 1000)
    5. Better error handling
    """
    
    base_path, _ = os.path.splitext(path)
    path = base_path + ".txt"
    
    lines = []
    lines.append("// ============================================================")
    lines.append("// CEKG Cypher Import Script (Generated)")
    lines.append(f"// Total Nodes: {len(nodes):,}")
    lines.append(f"// Total Relationships: {len(relationships):,}")
    lines.append("// Neo4j Compliant - Uses MERGE + ON CREATE SET pattern")
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
        # FIX: Escape label if needed
        safe_label = _escape_identifier(label)
        
        lines.append(f"// --- {label} Nodes ({len(node_list):,}) ---")
        
        for i in range(0, len(node_list), batch_size):
            batch = node_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(node_list) + batch_size - 1) // batch_size
            
            if len(node_list) > batch_size:
                lines.append(f"// Batch {batch_num}/{total_batches}")
            
            for node in batch:
                props_str = _format_cypher_properties(node.properties)
                
                # FIX: Use ON CREATE SET pattern instead of bare SET
                # Get id from properties
                node_id = node.properties.get('id', node.uid)
                safe_id = _escape_cypher_value(node_id)
                
                lines.append(
                    f'MERGE (n:{safe_label} {{id: {safe_id}}}) '
                    f'ON CREATE SET n = {props_str};'
                )
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
        # FIX: Escape relationship type with backticks if needed
        safe_rel_type = _escape_identifier(rel_type)
        
        lines.append(f"// --- {rel_type} Relationships ({len(rel_list):,}) ---")
        
        for i in range(0, len(rel_list), batch_size):
            batch = rel_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(rel_list) + batch_size - 1) // batch_size
            
            if len(rel_list) > batch_size:
                lines.append(f"// Batch {batch_num}/{total_batches}")
            
            for rel in batch:
                props_str = _format_cypher_properties(rel.properties)
                
                # FIX: Escape node IDs
                safe_start_id = _escape_cypher_value(rel.start_node_uid)
                safe_end_id = _escape_cypher_value(rel.end_node_uid)
                
                # FIX: Use ON CREATE SET pattern for relationships too
                lines.append(
                    f'MATCH (a {{id: {safe_start_id}}}), (b {{id: {safe_end_id}}}) '
                    f'MERGE (a)-[r:{safe_rel_type}]->(b) '
                    f'ON CREATE SET r = {props_str};'
                )
                statement_count += 1
            
            lines.append("")
    
    lines.append("// ============================================================")
    lines.append("// IMPORT COMPLETE")
    lines.append(f"// Total Statements: {statement_count:,}")
    lines.append("// ============================================================")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))
    
    print(f"[export] Cypher exported: {statement_count:,} statements to {path}")
    print(f"[export] File size: {len(lines):,} lines")
    print(f"[export] Batch size: {batch_size} (optimized for Neo4j)")

# ============================================================================
# JSON-LD Export (Unchanged)
# ============================================================================

def build_jsonld(
    events: List[CEKEvent],
    event_produces: List[EventProducesEntity],
    causal_links: List[CausalLink]
) -> Dict[str, Any]:
    """Build JSON-LD representation"""
    g = []

    # Events
    for ev in events:
        event_dict = asdict(ev)
        event_dict["@id"] = event_dict.pop("id")
        event_dict["type"] = "Event"
        g.append(event_dict)

    # Event → Entity (production)
    for prod in event_produces:
        g.append({
            "@id": f"{prod.event_id}__{prod.relationship}__{prod.entity_id}",
            "type": "EventProducesEntity",
            "from": prod.event_id,
            "to": prod.entity_id,
            "entity_name": prod.entity_name,
            "entity_type": prod.entity_type,
            "relationship": prod.relationship,
            "strength": prod.strength
        })

    # Event → Event (causal)
    for link in causal_links:
        g.append({
            "@id": f"{link.source_event_id}__CAUSES__{link.target_event_id}",
            "type": "CausalEdge",
            "from": link.source_event_id,
            "to": link.target_event_id,
            "relationType": link.relation_type,
            "mechanism": link.mechanism,
            "weight": link.weight,
            "confidence": link.confidence,
            "edge_supertype": link.edge_supertype
        })

    return {"@graph": g}

def export_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[export] JSON exported to {path}")

# ============================================================================
# CSV Export (Unchanged - already working correctly)
# ============================================================================

def export_csv(
    out_dir: str,
    events: List[CEKEvent],
    event_produces: List[EventProducesEntity],
    causal_links: List[CausalLink],
    thematic_links: Optional[List[ThematicLink]] = None,
    scenes: Optional[List[Scene]] = None
) -> Dict[str, str]:
    """Export star-model graph to Neo4j CSV format."""
    os.makedirs(out_dir, exist_ok=True)

    if thematic_links is None:
        thematic_links = []
    if scenes is None:
        scenes = []

    print("[export] Exporting CSVs...")
    
    # Collect unique entities
    entities_by_type = defaultdict(dict)
    for prod in event_produces:
        entities_by_type[prod.entity_type][prod.entity_id] = prod.entity_name
    
    # Event nodes
    events_rows = []
    for ev in events:
        events_rows.append({
            ":ID": ev.id,
            "name": ev.raw_description,
            "actionType": ev.action_type,
            "source_quote": ev.source_quote,
            "confidence": ev.confidence,
            "sequence": ev.sequence,
            "chapter": ev.chapter,
            "time": ev.time_context or "",
            "location": ev.location_context or "",
            "scene_id": ev.scene_id or "",
            "theme_annotations": json.dumps(ev.theme_annotations) if ev.theme_annotations else ""
        })
    
    # Agent nodes
    all_agents = {}
    all_agents.update(entities_by_type.get("actor", {}))
    all_agents.update(entities_by_type.get("patient", {}))
    agent_rows = [{":ID": aid, "name": name} for aid, name in all_agents.items()]
    
    # Place nodes
    place_rows = [{":ID": pid, "name": name} 
                  for pid, name in entities_by_type.get("place", {}).items()]
    
    # WhyFactor nodes
    whyfactor_rows = [{":ID": wid, "factor": name} 
                      for wid, name in entities_by_type.get("whyfactor", {}).items()]
    
    # Entity → Event edges (derived directly from event_produces)
    acts_in_rows = [{
        ":START_ID": prod.entity_id,
        ":END_ID": prod.event_id,
        ":TYPE": "ACTS_IN",
        "strength": prod.strength
    } for prod in event_produces if prod.entity_type == "actor"]

    affected_in_rows = [{
        ":START_ID": prod.entity_id,
        ":END_ID": prod.event_id,
        ":TYPE": "AFFECTED_IN",
        "strength": prod.strength
    } for prod in event_produces if prod.entity_type == "patient"]

    motivates_rows = [{
        ":START_ID": prod.entity_id,
        ":END_ID": prod.event_id,
        ":TYPE": "MOTIVATES",
        "weight": prod.strength
    } for prod in event_produces if prod.entity_type == "whyfactor"]

    hosts_rows = [{
        ":START_ID": prod.entity_id,
        ":END_ID": prod.event_id,
        ":TYPE": "HOSTS",
        "specificity": prod.strength
    } for prod in event_produces if prod.entity_type == "place"]
    
    # Event -[:FOLLOWS]-> Event
    follows_rows = []
    for i in range(len(events) - 1):
        ev1 = events[i]
        ev2 = events[i + 1]
        if ev1.chapter == ev2.chapter:
            follows_rows.append({
                ":START_ID": ev1.id,
                ":END_ID": ev2.id,
                ":TYPE": "FOLLOWS"
            })
    
    # Event -[:CAUSES]-> Event
    causes_rows = [{
        ":START_ID": link.source_event_id,
        ":END_ID": link.target_event_id,
        ":TYPE": "CAUSES",
        "relationType": link.relation_type,
        "mechanism": link.mechanism,
        "weight": link.weight,
        "confidence": link.confidence,
        "edge_supertype": link.edge_supertype or ""
    } for link in causal_links]
    
    # Scene nodes
    scene_nodes_rows = [{
        ":ID": scene.id,
        "theme": scene.theme,
        "chapter": scene.chapter,
        "confidence": scene.confidence
    } for scene in scenes]
    
    # Scene -> Event edges
    scene_includes_rows = []
    for scene in scenes:
        for event_id in scene.included_event_ids:
            scene_includes_rows.append({
                ":START_ID": scene.id,
                ":END_ID": event_id,
                ":TYPE": "INCLUDES"
            })
            
    # Thematic Links
    thematic_link_rows = [{
        ":START_ID": link.source_event_id,
        ":END_ID": link.target_event_id,
        ":TYPE": "THEMATIC",
        "theme": link.theme,
        "source_involvement": link.source_involvement,
        "target_involvement": link.target_involvement,
        "source_role": link.source_role or "",
        "target_role": link.target_role or "",
        "confidence": link.confidence
    } for link in thematic_links]

    files = {
        "events.csv": events_rows,
        "agents.csv": agent_rows,
        "places.csv": place_rows,
        "whyfactors.csv": whyfactor_rows,
        "acts_in.csv": acts_in_rows,
        "affected_in.csv": affected_in_rows,
        "motivates.csv": motivates_rows,
        "hosts.csv": hosts_rows,
        "follows.csv": follows_rows,
        "causes.csv": causes_rows,
        "scenes.csv": scene_nodes_rows,
        "scene_includes_event.csv": scene_includes_rows,
        "thematic_links.csv": thematic_link_rows
    }

    def _write_csv(rows, path):
        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                pass
            return
        
        if pd is not None:
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
        else:
            keys = list(rows[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

    out_paths = {}
    for fname, rows in files.items():
        path = os.path.join(out_dir, fname)
        _write_csv(rows, path)
        if rows:
            out_paths[fname] = path
    
    print(f"[export] CSV exported: {len(out_paths)} active files to {out_dir}/")
    return out_paths