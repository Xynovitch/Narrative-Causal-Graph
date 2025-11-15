import os
import json
import csv
from collections import defaultdict
from dataclasses import asdict
from typing import List, Dict, Any

try:
    import pandas as pd
except ImportError:
    pd = None

# --- FIX IS HERE (PART 1) ---
# We must import GenericNode and GenericRelationship directly
from .schemas import (
    CEKEvent, EventProducesEntity, EntityPointsToEvent, CausalLink,
    GenericNode, GenericRelationship
)
# --- END FIX ---

from .utils import _truncate_safe, _escape_cypher_string

def build_jsonld(
    events: List[CEKEvent],
    event_produces: List[EventProducesEntity],
    entity_points_to: List[EntityPointsToEvent],
    causal_links: List[CausalLink]
) -> Dict[str, Any]:
    """Build JSON-LD representation with DUAL FLOW structure"""
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
            "@id": f"{link.cause_id}__CAUSES__{link.effect_id}",
            "type": "CausalEdge",
            "from": link.cause_id,
            "to": link.effect_id,
            "relationType": link.relationType,
            "mechanism": link.mechanism,
            "sign": link.sign,
            "weight": link.weight,
            "confidence": link.confidence,
            "cause_sequence": link.cause_sequence,
            "effect_sequence": link.effect_sequence
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
            prop_list.append(f"{k}: \"{v}\"") # Assumes strings are pre-escaped
        elif isinstance(v, bool):
            prop_list.append(f"{k}: {str(v).lower()}")
        elif v is None:
            prop_list.append(f"{k}: null")
        else:
            prop_list.append(f"{k}: {v}")
    return f"{{{', '.join(prop_list)}}}"


def export_neo4j_cypher(
    path: str,
    # --- FIX IS HERE (PART 2) ---
    # Remove the 'schemas.' prefix because we imported the classes directly
    nodes: List[GenericNode],
    relationships: List[GenericRelationship]
    # --- END FIX ---
):
    """
    Export a generic graph to a Neo4j Cypher script.
    This function is now "dumb" and data-driven.
    """
    
    # Force the file to be saved as a .txt file
    base_path, _ = os.path.splitext(path)
    path = base_path + ".txt"
    
    lines = []
    lines.append("// ============================================================")
    lines.append("// CEKG Cypher Import Script (Generated)")
    lines.append("// ============================================================\n")
    
    # 1. Create Nodes
    lines.append("// 1. CREATE NODES")
    # Group nodes by label for cleaner output
    nodes_by_label = defaultdict(list)
    for node in nodes:
        nodes_by_label[node.label].append(node)
        
    for label, node_list in nodes_by_label.items():
        lines.append(f"// --- {label} Nodes ({len(node_list)}) ---")
        for node in node_list:
            props_str = _format_cypher_properties(node.properties)
            # Use MERGE on 'id' to make the script idempotent
            lines.append(f"MERGE (n:{label} {{id: \"{node.uid}\"}}) SET n = {props_str};")
        lines.append("")

    lines.append("// ============================================================")
    lines.append("// 2. CREATE RELATIONSHIPS")
    lines.append("// ============================================================\n")

    # 2. Create Relationships
    # Group relationships by type for cleaner output
    rels_by_type = defaultdict(list)
    for rel in relationships:
        rels_by_type[rel.rel_type].append(rel)

    for rel_type, rel_list in rels_by_type.items():
        lines.append(f"// --- {rel_type} Relationships ({len(rel_list)}) ---")
        for rel in rel_list:
            props_str = _format_cypher_properties(rel.properties)
            # Match on the unique IDs
            lines.append(
                f"MATCH (a {{id: \"{rel.start_node_uid}\"}}), (b {{id: \"{rel.end_node_uid}\"}}) "
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
    causal_links: List[CausalLink]
) -> Dict[str, str]:
    """Export DUAL FLOW structure to Neo4j CSV format"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Collect unique entities
    entities_by_type = defaultdict(dict)
    for prod in event_produces:
        entities_by_type[prod.entity_type][prod.entity_id] = prod.entity_name
    
    # Event nodes
    events_rows = []
    for ev in events:
        events_rows.append({
            ":ID": ev.id,
            "name": ev.name,
            "eventType": ev.eventType,
            "actionType": ev.actionType,
            "source_quote": ev.source_quote,
            "causeWeight": ev.causeWeight or 0.0,
            "confidence": ev.confidence,
            "sequence": ev.sequence,
            "chapter": ev.chapter,
            "time": ev.time or "",
            "location": ev.location or ""
        })
    
    # Agent nodes
    all_agents = {}
    all_agents.update(entities_by_type.get("actor", {}))
    all_agents.update(entities_by_type.get("patient", {}))
    agent_rows = [{":ID": aid, "name": name} for aid, name in all_agents.items()]
    
    # Place nodes
    place_rows = [{":ID": pid, "name": name} for pid, name in entities_by_type.get("place", {}).items()]
    
    # WhyFactor nodes
    whyfactor_rows = [{":ID": wid, "factor": name} for wid, name in entities_by_type.get("whyfactor", {}).items()]
    
    # Event → Entity edges
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
    
    # Entity → Event edges
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
        ":START_ID": link.cause_id,
        ":END_ID": link.effect_id,
        ":TYPE": "CAUSES",
        "relationType": link.relationType,
        "mechanism": link.mechanism,
        "sign": link.sign,
        "weight": link.weight,
        "confidence": link.confidence,
        "cause_seq": link.cause_sequence,
        "effect_seq": link.effect_sequence
    } for link in causal_links]

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
    }

    def _write_csv(rows, path):
        if not rows:
            # Create empty file for consistency
            open(path, "w", encoding="utf-8").close()
            return
        
        if pd is not None:
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
        else:
            # Fallback to csv module if pandas isn't installed
            keys = list(rows[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

    out_paths = {}
    for fname, rows in files.items():
        # Only write non-empty files for nodes/edges
        if rows: 
            path = os.path.join(out_dir, fname)
            _write_csv(rows, path)
            out_paths[fname] = path
    
    print(f"[export] CSV exported: {len(out_paths)} files to {out_dir}/")
    return out_paths