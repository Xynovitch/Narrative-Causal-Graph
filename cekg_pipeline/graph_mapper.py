from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from . import schemas
from . import utils

def _escape_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Escapes all string properties for Cypher."""
    escaped = {}
    for k, v in props.items():
        if isinstance(v, str):
            escaped[k] = utils._escape_cypher_string(v)
        else:
            escaped[k] = v
    return escaped

def _sanitize_name_for_id(name: str) -> str:
    """Creates a safe ID string from a name."""
    if not name: return "unknown"
    clean = name.lower().replace(' ', '_')
    clean = clean.replace('"', '').replace("'", "").replace(":", "").replace("\\", "")
    return clean

def map_to_generic_graph(
    events: List[schemas.CEKEvent],
    event_produces: List[schemas.EventProducesEntity],
    entity_points_to: List[schemas.EntityPointsToEvent],
    causal_links: List[schemas.CausalLink],
    graph_model: str = "star", # <--- New Argument
    semantic_links: Optional[List[schemas.SemanticLink]] = None,
    scenes: Optional[List[schemas.Scene]] = None
) -> Tuple[List[schemas.GenericNode], List[schemas.GenericRelationship]]:
    """
    Maps specific pipeline data into generic nodes and relationships.
    Handles the logic for Star vs Chain topology for Neo4j export.
    """
    nodes: Dict[str, schemas.GenericNode] = {}
    relationships: List[schemas.GenericRelationship] = []
    
    if semantic_links is None: semantic_links = []
    if scenes is None: scenes = []

    # 1. Map Events to Nodes
    for ev in events:
        nodes[ev.id] = schemas.GenericNode(
            uid=ev.id,
            label="Event",
            properties=_escape_props({
                "id": ev.id, "name": ev.raw_description, "eventType": "event",
                "category": ev.event_category, "actionType": ev.action_type,
                "confidence": ev.confidence, "chapter": ev.chapter, 
                "sequence": ev.sequence, "location": ev.location_context or "",
                "time": ev.time_context or "", "source_quote": utils._truncate_safe(ev.source_quote, 300)
            })
        )

    # 2. Map Entities to Nodes
    # In Star Mode, we create Canonical Nodes (Global). In Chain Mode, we create Event-Specific Nodes.
    entities_by_type = defaultdict(dict)
    for prod in event_produces:
        entities_by_type[prod.entity_type][prod.entity_id] = prod.entity_name

    # Consolidate Agents & WhyFactors
    all_entities = []
    for t in ['actor', 'patient', 'whyfactor']:
        for eid, ename in entities_by_type.get(t, {}).items():
            all_entities.append((eid, ename, t))

    for entity_id, entity_name, entity_type in all_entities:
        
        # Determine Node ID based on Graph Model
        if graph_model == "star":
            # Global Canonical ID
            safe_name = _sanitize_name_for_id(entity_name)
            if entity_type == 'whyfactor': safe_name = safe_name[:30]
            node_uid = f"{entity_type}_{safe_name}"
        else:
            # Event-Specific ID (Chain)
            node_uid = entity_id

        label = "Agent" if entity_type in ['actor', 'patient'] else "WhyFactor"
        
        if node_uid not in nodes:
            nodes[node_uid] = schemas.GenericNode(
                uid=node_uid,
                label=label,
                properties=_escape_props({"id": node_uid, "name": entity_name})
            )

    # 3. Map Scenes (New Feature)
    for scene in scenes:
        if scene.id not in nodes:
            nodes[scene.id] = schemas.GenericNode(
                uid=scene.id,
                label="Scene",
                properties=_escape_props({
                    "id": scene.id, "theme": scene.theme, 
                    "chapter": scene.chapter, "confidence": scene.confidence
                })
            )
        # Scene -> Event Relationships
        for event_id in scene.included_event_ids:
            if event_id in nodes:
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=scene.id, end_node_uid=event_id,
                    rel_type="INCLUDES", properties={}
                ))

    # 4. Map Relationships (Star vs Chain Logic)
    if graph_model == "star":
        # Star Mode: Canonical Entity -> ACTS_IN -> Event
        # We derived these links in graph_builder.create_entity_to_event_links
        # But we need to ensure the START node is the Canonical ID
        for link in entity_points_to:
            safe_name = _sanitize_name_for_id(link.entity_name)
            if link.entity_type == 'whyfactor': safe_name = safe_name[:30]
            canonical_uid = f"{link.entity_type}_{safe_name}"
            
            # Map canonical type to Agent/WhyFactor prefix if needed, 
            # but above we used "actor_pip". Let's stick to that consistency.
            # Actually, line 75 defined canonical_uid as "{entity_type}_{safe_name}".
            # We must match that exactly.
            # Note: 'patient' type is mapped to 'Agent' label but ID prefix should match.
            # To be safe, let's re-use the type from the link.
            
            # Special case: 'patient' and 'actor' both map to 'Agent' nodes, 
            # but we might want them to share the same node 'agent_pip'.
            prefix = "agent" if link.entity_type in ['actor', 'patient'] else link.entity_type
            canonical_uid = f"{prefix}_{safe_name}"

            if canonical_uid in nodes:
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=canonical_uid,
                    end_node_uid=link.next_event_id,
                    rel_type=link.relationship, # ACTS_IN, AFFECTED_IN, etc.
                    properties={"strength": link.strength}
                ))

    else:
        # Chain Mode: Event -> PRODUCES -> EntityInstance -> ACTS_IN -> NextEvent
        
        # 4a. Event -> Produces -> Entity
        for prod in event_produces:
            relationships.append(schemas.GenericRelationship(
                start_node_uid=prod.event_id,
                end_node_uid=prod.entity_id,
                rel_type=prod.relationship,
                properties={"strength": prod.strength}
            ))
            
        # 4b. Entity -> Acts_In -> NextEvent
        for link in entity_points_to:
            relationships.append(schemas.GenericRelationship(
                start_node_uid=link.entity_id,
                end_node_uid=link.next_event_id,
                rel_type=link.relationship,
                properties={"strength": link.strength}
            ))

    # 5. Causal Links (Same for both)
    for link in causal_links:
        relationships.append(schemas.GenericRelationship(
            start_node_uid=link.source_event_id,
            end_node_uid=link.target_event_id,
            rel_type=link.relation_type, # Now a dynamic string
            properties=_escape_props({
                "mechanism": utils._truncate_safe(link.mechanism, 200),
                "weight": link.weight,
                "confidence": link.confidence
            })
        ))

    # 6. Semantic Links (New Feature)
    for link in semantic_links:
        for source_id in link.source_event_ids:
            for target_id in link.target_event_ids:
                if source_id in nodes and target_id in nodes:
                    relationships.append(schemas.GenericRelationship(
                        start_node_uid=source_id,
                        end_node_uid=target_id,
                        rel_type=link.relation.upper(),
                        properties=_escape_props({"cue": str(link.cue), "confidence": link.confidence})
                    ))

    return list(nodes.values()), relationships