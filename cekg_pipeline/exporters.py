import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
from .schemas import CEKEvent, EventProducesEntity, EntityPointsToEvent, CausalLink, SemanticLink, Scene

def export_json(path: str, data: Dict[str, Any]) -> None:
    """
    Exports a dictionary to a JSON file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def export_csv(
    output_dir: str,
    events: List[CEKEvent],
    produces: List[EventProducesEntity],
    entity_links: List[EntityPointsToEvent],
    causal_links: List[CausalLink],
    semantic_links: Optional[List[SemanticLink]] = None,
    scenes: Optional[List[Scene]] = None
) -> Dict[str, str]:
    """
    Exports graph data to CSVs.
    Updated to reflect schemas.py changes (No Place/Time nodes).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    generated_files = {}

    # 1. EVENTS CSV (Updated Schema)
    events_data = []
    for e in events:
        events_data.append({
            "id": e.id,
            "raw_description": e.raw_description,
            "event_category": e.event_category,
            "action_type": e.action_type,
            "location_context": e.location_context, # Attribute
            "time_context": e.time_context,         # Attribute
            "chapter": e.chapter,
            "sequence": e.sequence,
            "confidence": e.confidence,
            "source_quote": e.source_quote
        })
    df_events = pd.DataFrame(events_data)
    path_events = os.path.join(output_dir, "events.csv")
    df_events.to_csv(path_events, index=False)
    generated_files["events"] = path_events

    # 2. PRODUCES CSV (Entity nodes: Actors/WhyFactors only)
    produces_data = []
    for p in produces:
        # Skip 'place' type if any remain (since they are now attributes)
        if p.entity_type != 'place': 
            produces_data.append({
                "event_id": p.event_id,
                "entity_id": p.entity_id,
                "entity_name": p.entity_name,
                "entity_type": p.entity_type,
                "relationship": p.relationship,
                "strength": p.strength
            })
    df_produces = pd.DataFrame(produces_data)
    path_produces = os.path.join(output_dir, "event_produces_entity.csv")
    df_produces.to_csv(path_produces, index=False)
    generated_files["produces"] = path_produces

    # 3. ENTITY LINKS CSV (Actor -> Event)
    links_data = [
        {
            "entity_id": l.entity_id,
            "entity_name": l.entity_name,
            "entity_type": l.entity_type,
            "next_event_id": l.next_event_id,
            "relationship": l.relationship,
            "strength": l.strength
        }
        for l in entity_links
    ]
    df_links = pd.DataFrame(links_data)
    path_links = os.path.join(output_dir, "entity_points_to_event.csv")
    df_links.to_csv(path_links, index=False)
    generated_files["entity_links"] = path_links

    # 4. CAUSAL LINKS CSV
    causal_data = [
        {
            "source": l.source_event_id,
            "target": l.target_event_id,
            "type": l.relation_type,
            "mechanism": l.mechanism,
            "weight": l.weight,
            "confidence": l.confidence
        }
        for l in causal_links
    ]
    df_causal = pd.DataFrame(causal_data)
    path_causal = os.path.join(output_dir, "causal_links.csv")
    df_causal.to_csv(path_causal, index=False)
    generated_files["causal"] = path_causal
    
    # 5. SEMANTIC LINKS (Optional)
    if semantic_links:
        sem_data = [
            {
                "source": l.source_event_ids[0] if l.source_event_ids else "",
                "target": l.target_event_ids[0] if l.target_event_ids else "",
                "relation": l.relation,
                "confidence": l.confidence
            }
            for l in semantic_links
        ]
        df_sem = pd.DataFrame(sem_data)
        path_sem = os.path.join(output_dir, "semantic_links.csv")
        df_sem.to_csv(path_sem, index=False)
        generated_files["semantic"] = path_sem

    return generated_files

def build_jsonld(
    events: List[CEKEvent],
    produces: List[EventProducesEntity],
    entity_links: List[EntityPointsToEvent],
    causal_links: List[CausalLink]
) -> Dict[str, Any]:
    """
    Constructs a JSON-LD style dictionary.
    """
    graph_nodes = []
    graph_edges = []
    
    # Events
    for e in events:
        graph_nodes.append({
            "id": e.id,
            "type": "Event",
            "properties": {
                "raw_description": e.raw_description,
                "category": e.event_category,
                "location": e.location_context,
                "time": e.time_context,
                "chapter": e.chapter,
                "sequence": e.sequence
            }
        })

    # Causal Edges
    for l in causal_links:
        graph_edges.append({
            "source": l.source_event_id,
            "target": l.target_event_id,
            "label": l.relation_type,
            "properties": {"mechanism": l.mechanism}
        })
        
    return {"nodes": graph_nodes, "edges": graph_edges}

def export_neo4j_cypher(
    output_path: str,
    events: List[CEKEvent],
    produces: List[EventProducesEntity],
    entity_links: List[EntityPointsToEvent],
    causal_links: List[CausalLink],
    semantic_links: Optional[List[SemanticLink]] = None,
    scenes: Optional[List[Scene]] = None
) -> None:
    """
    Generates a generic .cypher script to recreate the graph structure in Neo4j.
    Uses UNWIND batches for efficient importing.
    """
    
    def clean(s):
        if s is None: return ""
        return str(s).replace("\\", "\\\\").replace("'", "\\'")

    queries = []

    # 1. CONSTRAINTS
    queries.append("// --- Constraints ---")
    queries.append("CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;")
    queries.append("CREATE CONSTRAINT character_id_unique IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE;")
    queries.append("CREATE CONSTRAINT why_id_unique IF NOT EXISTS FOR (w:WhyFactor) REQUIRE w.id IS UNIQUE;")
    queries.append("CREATE CONSTRAINT scene_id_unique IF NOT EXISTS FOR (s:Scene) REQUIRE s.id IS UNIQUE;")
    queries.append("")

    # 2. NODES: EVENTS
    queries.append("// --- Create Events ---")
    event_maps = []
    for e in events:
        event_maps.append(
            f"{{id: '{clean(e.id)}', name: '{clean(e.raw_description)}', "
            f"category: '{clean(e.event_category)}', "
            f"location: '{clean(e.location_context)}', time: '{clean(e.time_context)}', "
            f"chapter: {e.chapter}, sequence: {e.sequence}, "
            f"quote: '{clean(e.source_quote)}'}}"
        )
    
    chunk_size = 500
    for i in range(0, len(event_maps), chunk_size):
        chunk = event_maps[i:i+chunk_size]
        queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
        queries.append("MERGE (e:Event {id: row.id})")
        queries.append("SET e += row;")
    queries.append("")

    # 3. NODES: ENTITIES
    queries.append("// --- Create Entities ---")
    actors = {}
    whyfactors = {}
    
    for p in produces:
        if p.entity_type in ['actor', 'patient']:
            actors[p.entity_id] = p.entity_name
        elif p.entity_type == 'whyfactor':
            whyfactors[p.entity_id] = p.entity_name
            
    # Actors
    if actors:
        actor_maps = [f"{{id: '{clean(k)}', name: '{clean(v)}'}}" for k, v in actors.items()]
        for i in range(0, len(actor_maps), chunk_size):
            chunk = actor_maps[i:i+chunk_size]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append("MERGE (c:Character {id: row.id})")
            queries.append("SET c.name = row.name;")

    # WhyFactors
    if whyfactors:
        why_maps = [f"{{id: '{clean(k)}', name: '{clean(v)}'}}" for k, v in whyfactors.items()]
        for i in range(0, len(why_maps), chunk_size):
            chunk = why_maps[i:i+chunk_size]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append("MERGE (w:WhyFactor {id: row.id})")
            queries.append("SET w.name = row.name;")
    queries.append("")
    
    # 4. NODES: SCENES
    if scenes:
        queries.append("// --- Create Scenes ---")
        scene_maps = []
        for s in scenes:
            scene_maps.append(
                f"{{id: '{clean(s.id)}', theme: '{clean(s.theme)}', "
                f"chapter: {s.chapter}, confidence: {s.confidence}}}"
            )
        for i in range(0, len(scene_maps), chunk_size):
            chunk = scene_maps[i:i+chunk_size]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append("MERGE (s:Scene {id: row.id})")
            queries.append("SET s += row;")
        queries.append("")

    # 5. CAUSAL LINKS
    queries.append("// --- Create Causal Links ---")
    links_by_type = defaultdict(list)
    for l in causal_links:
        rel = l.relation_type if l.relation_type else "CAUSES"
        links_by_type[rel].append(l)
        
    for rel_type, links in links_by_type.items():
        type_maps = [
            f"{{source: '{clean(l.source_event_id)}', target: '{clean(l.target_event_id)}', "
            f"mechanism: '{clean(l.mechanism)}'}}" 
            for l in links
        ]
        for i in range(0, len(type_maps), chunk_size):
            chunk = type_maps[i:i+chunk_size]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append(f"MATCH (s:Event {{id: row.source}}), (t:Event {{id: row.target}})")
            queries.append(f"MERGE (s)-[r:{rel_type}]->(t)")
            queries.append("SET r.mechanism = row.mechanism;")
    queries.append("")

    # 6. EVENT PRODUCES ENTITY
    queries.append("// --- Event Produces Entity ---")
    prod_by_rel = defaultdict(list)
    for p in produces:
        target_label = "Character" if p.entity_type in ['actor', 'patient'] else "WhyFactor"
        if p.entity_type == 'place': continue
        key = (p.relationship or "RELATED", target_label)
        prod_by_rel[key].append(p)
        
    for (rel_type, target_label), items in prod_by_rel.items():
        maps = [f"{{evt: '{clean(x.event_id)}', ent: '{clean(x.entity_id)}'}}" for x in items]
        for i in range(0, len(maps), chunk_size):
            chunk = maps[i:i+chunk_size]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append(f"MATCH (e:Event {{id: row.evt}}), (t:{target_label} {{id: row.ent}})")
            queries.append(f"MERGE (e)-[:{rel_type}]->(t);")

    # 7. ENTITY ACTS IN
    queries.append("")
    queries.append("// --- Entity Acts In ---")
    ent_links_by_rel = defaultdict(list)
    for l in entity_links:
        source_label = "Character" if l.entity_type in ['actor', 'patient'] else "WhyFactor"
        key = (l.relationship, source_label)
        ent_links_by_rel[key].append(l)

    for (rel_type, source_label), items in ent_links_by_rel.items():
        maps = [f"{{ent: '{clean(x.entity_id)}', evt: '{clean(x.next_event_id)}'}}" for x in items]
        for i in range(0, len(maps), chunk_size):
            chunk = maps[i:i+chunk_size]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append(f"MATCH (s:{source_label} {{id: row.ent}}), (t:Event {{id: row.evt}})")
            queries.append(f"MERGE (s)-[:{rel_type}]->(t);")

    # 8. SCENE INCLUDES
    if scenes:
        queries.append("")
        queries.append("// --- Scene Includes ---")
        scene_links = []
        for s in scenes:
            for eid in s.included_event_ids:
                scene_links.append(f"{{scene: '{clean(s.id)}', evt: '{clean(eid)}'}}")
        
        for i in range(0, len(scene_links), chunk_size):
            chunk = scene_links[i:i+chunk_size]
            queries.append(f"UNWIND [{', '.join(chunk)}] AS row")
            queries.append("MATCH (s:Scene {id: row.scene}), (e:Event {id: row.evt})")
            queries.append("MERGE (s)-[:INCLUDES]->(e);")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(queries))