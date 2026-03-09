"""
Updated Graph Mapper with Scene-Centric Structure

Key Changes:
1. All events belong to scenes
2. All entities belong to scenes (through events)
3. Supports agent type classification
4. Handles mixed theory graphs
"""

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
    """
    Creates a safe ID string from a name for canonical IDs.
    """
    if not name:
        return "unknown"
    clean = name.lower().replace(' ', '_')
    clean = clean.replace('"', '').replace("'", "").replace(":", "").replace("\\", "")
    return clean

def map_to_generic_graph(
    events: List[schemas.CEKEvent],
    event_produces: List[schemas.EventProducesEntity],
    entity_points_to: List[schemas.EntityPointsToEvent],
    causal_links: List[schemas.CausalLink],
    graph_model: str = "star",
    semantic_links: Optional[List[schemas.SemanticLink]] = None,
    scenes: Optional[List[schemas.Scene]] = None,
    agent_classifications: Optional[Dict[str, str]] = None
) -> Tuple[List[schemas.GenericNode], List[schemas.GenericRelationship]]:
    """
    Maps specific pipeline data into generic nodes and relationships.
    Scene-centric: all entities and events are linked to scenes.
    """
    nodes: Dict[str, schemas.GenericNode] = {}
    relationships: List[schemas.GenericRelationship] = []
    
    if semantic_links is None:
        semantic_links = []
    if scenes is None:
        scenes = []
    if agent_classifications is None:
        agent_classifications = {}

    print(f"[graph_mapper] Mapping to generic graph (model: {graph_model})")
    print(f"[graph_mapper] Scene-centric structure with {len(scenes)} scenes")

    # 1. Map Events to Nodes
    for ev in events:
        nodes[ev.id] = schemas.GenericNode(
            uid=ev.id,
            label="Event",
            properties=_escape_props({
                "id": ev.id,
                "name": ev.raw_description,
                "eventType": "event",
                "category": ev.event_category,
                "actionType": ev.action_type,
                "theory": ev.theory,
                "confidence": ev.confidence,
                "chapter": ev.chapter,
                "sequence": ev.sequence,
                "location": ev.location_context or "",
                "time": ev.time_context or "",
                "source_quote": utils._truncate_safe(ev.source_quote, 300)
            })
        )

    # 2. Map Entities to Nodes (with agent types if available)
    entities_by_type = defaultdict(dict)
    for prod in event_produces:
        entities_by_type[prod.entity_type][prod.entity_id] = {
            "name": prod.entity_name,
            "agent_type": prod.agent_type,
            "theory": prod.theory
        }

    # Consolidate Agents & WhyFactors
    all_entities = []
    for t in ['actor', 'patient', 'whyfactor']:
        for eid, edata in entities_by_type.get(t, {}).items():
            all_entities.append((eid, edata, t))

    for entity_id, entity_data, entity_type in all_entities:
        entity_name = entity_data["name"]
        agent_type = entity_data.get("agent_type")
        theory = entity_data.get("theory", "@McKee")
        
        # Determine Node ID based on Graph Model
        if graph_model == "star":
            # Global Canonical ID
            safe_name = _sanitize_name_for_id(entity_name)
            if entity_type == 'whyfactor':
                safe_name = safe_name[:30]
            
            prefix = "agent" if entity_type in ['actor', 'patient'] else entity_type
            node_uid = f"{prefix}_{safe_name}"
        else:
            # Event-Specific ID (Chain)
            node_uid = entity_id

        label = "Agent" if entity_type in ['actor', 'patient'] else "WhyFactor"
        
        if node_uid not in nodes:
            props = {
                "id": node_uid,
                "name": entity_name
            }
            
            # Add agent type if classified
            if agent_type:
                props["agentType"] = agent_type
                props["theory"] = theory
            
            nodes[node_uid] = schemas.GenericNode(
                uid=node_uid,
                label=label,
                properties=_escape_props(props)
            )

    # 3. Map Scenes (CRITICAL: All events and entities belong to scenes)
    event_to_scene = {}  # Track which scene each event belongs to
    
    for scene in scenes:
        if scene.id not in nodes:
            nodes[scene.id] = schemas.GenericNode(
                uid=scene.id,
                label="Scene",
                properties=_escape_props({
                    "id": scene.id,
                    "theme": scene.theme,
                    "chapter": scene.chapter,
                    "confidence": scene.confidence,
                    "location": scene.primary_location or "",
                    "time": scene.time_period or "",
                    "place_type": scene.place_type or "",
                    "time_type": scene.time_type or ""
                })
            )
        
        # Scene -> Event Relationships
        for event_id in scene.included_event_ids:
            if event_id in nodes:
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=scene.id,
                    end_node_uid=event_id,
                    rel_type="INCLUDES",
                    properties={}
                ))
                event_to_scene[event_id] = scene.id
        
        # Scene -> Agent Relationships (all participants in scene)
        # Use the extended entity lists from scene
        all_scene_participants = set()
        if hasattr(scene, 'all_actors'):
            all_scene_participants.update(scene.all_actors)
        if hasattr(scene, 'all_patients'):
            all_scene_participants.update(scene.all_patients)
        
        for participant_name in all_scene_participants:
            safe_name = _sanitize_name_for_id(participant_name)
            agent_uid = f"agent_{safe_name}"
            
            if agent_uid in nodes:
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=scene.id,
                    end_node_uid=agent_uid,
                    rel_type="HAS_PARTICIPANT",
                    properties={}
                ))
        
        # Scene -> WhyFactor Relationships
        if hasattr(scene, 'all_whyfactors'):
            for whyfactor_name in scene.all_whyfactors:
                safe_name = _sanitize_name_for_id(whyfactor_name[:30])
                why_uid = f"whyfactor_{safe_name}"
                
                if why_uid in nodes:
                    relationships.append(schemas.GenericRelationship(
                        start_node_uid=scene.id,
                        end_node_uid=why_uid,
                        rel_type="HAS_MOTIVATION",
                        properties={}
                    ))

    # 4. Map Relationships (Star vs Chain Logic)
    if graph_model == "star":
        print("[graph_mapper] Using 'star' model (Canonical Entity -> Events)")
        # Star Mode: Canonical Entity -> ACTS_IN -> Event
        for link in entity_points_to:
            safe_name = _sanitize_name_for_id(link.entity_name)
            if link.entity_type == 'whyfactor':
                safe_name = safe_name[:30]
            
            prefix = "agent" if link.entity_type in ['actor', 'patient'] else link.entity_type
            canonical_uid = f"{prefix}_{safe_name}"

            if canonical_uid in nodes:
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=canonical_uid,
                    end_node_uid=link.next_event_id,
                    rel_type=link.relationship,
                    properties={"strength": link.strength}
                ))
    else:
        print("[graph_mapper] Using 'chain' model (Event->Entity->Event)")
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

    # 5. Map Event-Event (Follows) Relationships
    for i in range(len(events) - 1):
        ev1 = events[i]
        ev2 = events[i + 1]
        if ev1.chapter == ev2.chapter:
            relationships.append(schemas.GenericRelationship(
                start_node_uid=ev1.id,
                end_node_uid=ev2.id,
                rel_type="FOLLOWS",
                properties={}
            ))

    # 6. Causal Links (Mixed Theory - includes both McKee and Truby)
    for link in causal_links:
        relationships.append(schemas.GenericRelationship(
            start_node_uid=link.source_event_id,
            end_node_uid=link.target_event_id,
            rel_type=link.relation_type,
            properties=_escape_props({
                "mechanism": utils._truncate_safe(link.mechanism, 200),
                "weight": link.weight,
                "confidence": link.confidence,
                "theory": link.theory,
                "directionality": link.directionality
            })
        ))

    # 7. Semantic Links
    for link in semantic_links:
        for source_id in link.source_event_ids:
            for target_id in link.target_event_ids:
                if source_id in nodes and target_id in nodes:
                    relationships.append(schemas.GenericRelationship(
                        start_node_uid=source_id,
                        end_node_uid=target_id,
                        rel_type=link.relation.upper(),
                        properties=_escape_props({
                            "cue": str(link.cue) if link.cue else "",
                            "confidence": link.confidence
                        })
                    ))

    print(f"[graph_mapper] Created {len(nodes)} nodes and {len(relationships)} relationships")
    print(f"[graph_mapper] Scene-centric structure: {len(scenes)} scenes containing {len(events)} events")
    
    # Verify all events belong to scenes
    orphan_events = [e.id for e in events if e.id not in event_to_scene]
    if orphan_events:
        print(f"[warning] Found {len(orphan_events)} orphan events not in any scene")
    
    return list(nodes.values()), relationships