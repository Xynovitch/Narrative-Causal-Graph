import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
from .schemas import CEKEvent, EventProducesEntity, EntityPointsToEvent, CausalLink, SemanticLink, Scene, GenericNode, GenericRelationship

def export_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def export_csv(
    output_dir: str,
    events: List[CEKEvent],
    produces: List[EventProducesEntity],
    entity_links: List[EntityPointsToEvent],
    causal_links: List[CausalLink],
    semantic_links: Optional[List[SemanticLink]] = None,
    scenes: Optional[List[Scene]] = None,
    graph_model: str = "star"
) -> Dict[str, str]:
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    generated_files = {}

    # 1. EVENTS
    events_data = []
    for e in events:
        events_data.append({
            "id": e.id, "raw_description": e.raw_description, "event_category": e.event_category,
            "action_type": e.action_type, "location_context": e.location_context, "time_context": e.time_context,
            "chapter": e.chapter, "sequence": e.sequence, "confidence": e.confidence, "source_quote": e.source_quote
        })
    pd.DataFrame(events_data).to_csv(os.path.join(output_dir, "events.csv"), index=False)
    generated_files["events"] = "events.csv"

    # 2. ENTITIES (Nodes)
    actors = {}
    whyfactors = {}
    for p in produces:
        if p.entity_type in ['actor', 'patient']: actors[p.entity_id] = p.entity_name
        elif p.entity_type == 'whyfactor': whyfactors[p.entity_id] = p.entity_name
            
    if actors:
        pd.DataFrame([{"id": k, "name": v} for k, v in actors.items()]).to_csv(os.path.join(output_dir, "actors.csv"), index=False)
        generated_files["actors"] = "actors.csv"
    if whyfactors:
        pd.DataFrame([{"id": k, "name": v} for k, v in whyfactors.items()]).to_csv(os.path.join(output_dir, "whyfactors.csv"), index=False)
        generated_files["whyfactors"] = "whyfactors.csv"

    # 3. PRODUCES EDGES (Skip in Star Mode)
    if graph_model != "star":
        produces_data = []
        for p in produces:
            if p.entity_type != 'place': 
                produces_data.append({
                    "event_id": p.event_id, "entity_id": p.entity_id,
                    "entity_type": p.entity_type, "relationship": p.relationship, "strength": p.strength
                })
        if produces_data:
            pd.DataFrame(produces_data).to_csv(os.path.join(output_dir, "event_produces_entity.csv"), index=False)
            generated_files["produces"] = "event_produces_entity.csv"

    # 4. ENTITY LINKS (Entity -> Event)
    links_data = [{
        "entity_id": l.entity_id, "next_event_id": l.next_event_id,
        "relationship": l.relationship, "strength": l.strength
    } for l in entity_links]
    if links_data:
        pd.DataFrame(links_data).to_csv(os.path.join(output_dir, "entity_acts_in_event.csv"), index=False)
        generated_files["entity_links"] = "entity_acts_in_event.csv"

    # 5. CAUSAL LINKS
    causal_data = [{
        "source": l.source_event_id, "target": l.target_event_id, "type": l.relation_type,
        "mechanism": l.mechanism, "weight": l.weight, "confidence": l.confidence
    } for l in causal_links]
    if causal_data:
        pd.DataFrame(causal_data).to_csv(os.path.join(output_dir, "causal_links.csv"), index=False)
        generated_files["causal"] = "causal_links.csv"
    
    # 6. SCENES & SEMANTIC
    if scenes:
        pd.DataFrame([{"id": s.id, "theme": s.theme, "chapter": s.chapter} for s in scenes]).to_csv(os.path.join(output_dir, "scenes.csv"), index=False)
        scene_edges = []
        for s in scenes:
            for eid in s.included_event_ids:
                scene_edges.append({"scene_id": s.id, "event_id": eid})
        pd.DataFrame(scene_edges).to_csv(os.path.join(output_dir, "scene_includes_event.csv"), index=False)

    if semantic_links:
        pd.DataFrame([{"source": l.source_event_ids[0], "target": l.target_event_ids[0], "relation": l.relation} for l in semantic_links]).to_csv(os.path.join(output_dir, "semantic_links.csv"), index=False)

    return generated_files

def build_jsonld(events, produces, entity_links, causal_links):
    nodes = []
    edges = []
    for e in events:
        nodes.append({"id": e.id, "type": "Event", "properties": {"name": e.raw_description, "category": e.event_category}})
    for l in causal_links:
        edges.append({"source": l.source_event_id, "target": l.target_event_id, "label": l.relation_type})
    return {"nodes": nodes, "edges": edges}

def export_neo4j_cypher(output_path, nodes, relationships):
    def clean(s): return str(s).replace("\\", "\\\\").replace("'", "\\'") if s else ""
    queries = ["CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;", ""]

    nodes_by_label = defaultdict(list)
    for n in nodes: nodes_by_label[n.label].append(n)
        
    for label, node_list in nodes_by_label.items():
        node_maps = []
        for n in node_list:
            props = [f"{k}: '{clean(v)}'" if isinstance(v, str) else f"{k}: {v}" for k, v in n.properties.items()]
            node_maps.append(f"{{ {', '.join(props)} }}")
        
        for i in range(0, len(node_maps), 500):
            chunk = node_maps[i:i+500]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append(f"MERGE (n:{label} {{id: row.id}}) SET n += row;")
        queries.append("")

    rels_by_type = defaultdict(list)
    for r in relationships: rels_by_type[r.rel_type].append(r)
        
    for rel_type, rel_list in rels_by_type.items():
        rel_maps = []
        for r in rel_list:
            props = [f"{k}: '{clean(v)}'" if isinstance(v, str) else f"{k}: {v}" for k, v in r.properties.items()]
            rel_maps.append(f"{{start: '{clean(r.start_node_uid)}', end: '{clean(r.end_node_uid)}', {', '.join(props)}}}")
            
        for i in range(0, len(rel_maps), 500):
            chunk = rel_maps[i:i+500]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append(f"MATCH (a {{id: row.start}}), (b {{id: row.end}})")
            queries.append(f"MERGE (a)-[r:{rel_type}]->(b)")
            if any(k not in ['start', 'end'] for k in r.properties):
                 queries.append(f"SET r += row") 
        queries.append("")

    with open(output_path, 'w', encoding='utf-8') as f: f.write("\n".join(queries))