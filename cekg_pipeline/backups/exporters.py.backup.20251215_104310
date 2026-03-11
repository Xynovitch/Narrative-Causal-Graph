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
    CEKEvent, EventProducesEntity, EntityPointsToEvent, CausalLink,
    GenericNode, GenericRelationship, Scene, SemanticLink
)
from .utils import _truncate_safe, _escape_cypher_string

def build_jsonld(
    events: List[CEKEvent],
    event_produces: List[EventProducesEntity],
    entity_points_to: List[EntityPointsToEvent],
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
    
    # Entity → Event (pointing to next)
    for ept in entity_points_to:
        g.append({
            "@id": f"{ept.entity_id}__{ept.relationship}__{ept.next_event_id}",
            "type": "EntityPointsToEvent",
            "from": ept.entity_id,
            "to": ept.next_event_id,
            "entity_name": ept.entity_name,
            "entity_type": ept.entity_type,
            "relationship": ept.relationship,
            "strength": ept.strength
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
            "confidence": link.confidence
        })
    
    return {"@graph": g}

def export_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[export] JSON exported to {path}")


def _format_cypher_properties(props: Dict[str, Any]) -> str:
    """Helper to format a properties dictionary into a Cypher string."""
    prop_list = []
    for k, v in props.items():
        if isinstance(v, str):
            prop_list.append(f"{k}: \"{v}\"") 
        elif isinstance(v, bool):
            prop_list.append(f"{k}: {str(v).lower()}")
        elif v is None:
            prop_list.append(f"{k}: null")
        else:
            prop_list.append(f"{k}: {v}")
    return f"{{{', '.join(prop_list)}}}"


def export_neo4j_cypher(
    path: str,
    nodes: List[GenericNode],
    relationships: List[GenericRelationship]
):
    """Export a generic graph to a Neo4j Cypher script."""
    
    base_path, _ = os.path.splitext(path)
    path = base_path + ".txt"
    
    lines = []
    lines.append("// ============================================================")
    lines.append("// CEKG Cypher Import Script (Generated)")
    lines.append("// ============================================================\n")
    
    # 1. Create Nodes
    lines.append("// 1. CREATE NODES")
    nodes_by_label = defaultdict(list)
    for node in nodes:
        nodes_by_label[node.label].append(node)
        
    for label, node_list in nodes_by_label.items():
        lines.append(f"// --- {label} Nodes ({len(node_list)}) ---")
        for node in node_list:
            props_str = _format_cypher_properties(node.properties)
            safe_uid = _escape_cypher_string(node.uid)
            lines.append(f"MERGE (n:{label} {{id: \"{safe_uid}\"}}) SET n = {props_str};")
        lines.append("")

    lines.append("// ============================================================")
    lines.append("// 2. CREATE RELATIONSHIPS")
    lines.append("// ============================================================\n")

    # 2. Create Relationships
    rels_by_type = defaultdict(list)
    for rel in relationships:
        rels_by_type[rel.rel_type].append(rel)

    for rel_type, rel_list in rels_by_type.items():
        lines.append(f"// --- {rel_type} Relationships ({len(rel_list)}) ---")
        for rel in rel_list:
            props_str = _format_cypher_properties(rel.properties)
            safe_start_uid = _escape_cypher_string(rel.start_node_uid)
            safe_end_uid = _escape_cypher_string(rel.end_node_uid)
            lines.append(
                f"MATCH (a {{id: \"{safe_start_uid}\"}}), (b {{id: \"{safe_end_uid}\"}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) SET r = {props_str};"
            )
        lines.append("")
    
    lines.append("// ============================================================")
    lines.append("// SCRIPT COMPLETE")
    lines.append(f"// Total Nodes: {len(nodes)}")
    lines.append(f"// Total Relationships: {len(relationships)}")
    lines.append("// ============================================================")

    with open(path, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))
    
    print(f"[export] Cypher exported: {len(lines)} statements to {path}")


def export_csv(
    out_dir: str,
    events: List[CEKEvent],
    event_produces: List[EventProducesEntity],
    entity_points_to: List[EntityPointsToEvent],
    causal_links: List[CausalLink],
    semantic_links: Optional[List[SemanticLink]] = None,
    scenes: Optional[List[Scene]] = None,
    graph_model: str = "star"
) -> Dict[str, str]:
    """
    Export DUAL FLOW structure to Neo4j CSV format.
    Now supports conditional edge generation based on graph_model.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if semantic_links is None:
        semantic_links = []
    if scenes is None:
        scenes = []
    
    print(f"[export] Exporting CSVs (graph_model: {graph_model})...")
    
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
            "event_category": ev.event_category,
            "actionType": ev.action_type,
            "source_quote": ev.source_quote,
            "confidence": ev.confidence,
            "sequence": ev.sequence,
            "chapter": ev.chapter,
            "time": ev.time_context or "",
            "location": ev.location_context or ""
        })
    
    # Agent nodes
    all_agents = {}
    all_agents.update(entities_by_type.get("actor", {}))
    all_agents.update(entities_by_type.get("patient", {}))
    agent_rows = [{":ID": aid, "name": name} for aid, name in all_agents.items()]
    
    # Place nodes (kept for compatibility, but typically empty)
    place_rows = [{":ID": pid, "name": name} 
                  for pid, name in entities_by_type.get("place", {}).items()]
    
    # WhyFactor nodes
    whyfactor_rows = [{":ID": wid, "factor": name} 
                      for wid, name in entities_by_type.get("whyfactor", {}).items()]
    
    # --- CONDITIONAL EDGE GENERATION BASED ON GRAPH MODEL ---
    produces_actor_rows = []
    produces_patient_rows = []
    produces_motivation_rows = []
    produces_location_rows = []

    # Only generate "PRODUCES" edges if NOT in Star Mode
    if graph_model != "star":
        produces_actor_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_ACTOR",
            "strength": prod.strength
        } for prod in event_produces if prod.entity_type == "actor"]
        
        produces_patient_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_PATIENT",
            "strength": prod.strength
        } for prod in event_produces if prod.entity_type == "patient"]
        
        produces_motivation_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_MOTIVATION",
            "weight": prod.strength
        } for prod in event_produces if prod.entity_type == "whyfactor"]
        
        produces_location_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_LOCATION",
            "specificity": prod.strength
        } for prod in event_produces if prod.entity_type == "place"]
    
    # Entity → Event edges (Always generated, but represent different things)
    # Star Mode: Canonical Agent -> Event
    # Chain Mode: Agent Instance -> Next Event
    acts_in_rows = [{
        ":START_ID": ept.entity_id,
        ":END_ID": ept.next_event_id,
        ":TYPE": "ACTS_IN",
        "strength": ept.strength
    } for ept in entity_points_to if ept.relationship == "ACTS_IN"]
    
    affected_in_rows = [{
        ":START_ID": ept.entity_id,
        ":END_ID": ept.next_event_id,
        ":TYPE": "AFFECTED_IN",
        "strength": ept.strength
    } for ept in entity_points_to if ept.relationship == "AFFECTED_IN"]
    
    motivates_rows = [{
        ":START_ID": ept.entity_id,
        ":END_ID": ept.next_event_id,
        ":TYPE": "MOTIVATES",
        "weight": ept.strength
    } for ept in entity_points_to if ept.relationship == "MOTIVATES"]
    
    hosts_rows = [{
        ":START_ID": ept.entity_id,
        ":END_ID": ept.next_event_id,
        ":TYPE": "HOSTS",
        "specificity": ept.strength
    } for ept in entity_points_to if ept.relationship == "HOSTS"]
    
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
        "confidence": link.confidence
    } for link in causal_links]
    
    # Scene nodes (NEW)
    scene_nodes_rows = [{
        ":ID": scene.id,
        "theme": scene.theme,
        "chapter": scene.chapter,
        "confidence": scene.confidence
    } for scene in scenes]
    
    # Scene -> Event edges (NEW)
    scene_includes_rows = []
    for scene in scenes:
        for event_id in scene.included_event_ids:
            scene_includes_rows.append({
                ":START_ID": scene.id,
                ":END_ID": event_id,
                ":TYPE": "INCLUDES"
            })
            
    # Semantic Links (NEW)
    semantic_link_rows = []
    for link in semantic_links:
        for source_id in link.source_event_ids:
            for target_id in link.target_event_ids:
                semantic_link_rows.append({
                    ":START_ID": source_id,
                    ":END_ID": target_id,
                    ":TYPE": link.relation.upper(),
                    "cue": str(link.cue) if link.cue else "",
                    "confidence": link.confidence
                })

    files = {
        "events.csv": events_rows,
        "agents.csv": agent_rows,
        "places.csv": place_rows,
        "whyfactors.csv": whyfactor_rows,
        "produces_actor.csv": produces_actor_rows,
        "produces_patient.csv": produces_patient_rows,
        "produces_motivation.csv": produces_motivation_rows,
        "produces_location.csv": produces_location_rows,
        "acts_in.csv": acts_in_rows,
        "affected_in.csv": affected_in_rows,
        "motivates.csv": motivates_rows,
        "hosts.csv": hosts_rows,
        "follows.csv": follows_rows,
        "causes.csv": causes_rows,
        "scenes.csv": scene_nodes_rows,
        "scene_includes_event.csv": scene_includes_rows,
        "semantic_links.csv": semantic_link_rows
    }

    def _write_csv(rows, path):
        if not rows:
            # Overwrite with empty file if list is empty (prevents stale data)
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