from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import re  # Added for sanitization
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
    Creates a safe ID string from a name.
    1. Lowercase
    2. Replace spaces with underscores
    3. Remove quotes, colons, and other problematic chars
    """
    if not name:
        return "unknown"
    # Replace spaces with _
    clean = name.lower().replace(' ', '_')
    # Remove quotes and typical illegal chars
    clean = clean.replace('"', '').replace("'", "").replace(":", "").replace("\\", "")
    return clean

def map_to_generic_graph(
    events: List[schemas.CEKEvent],
    event_produces: List[schemas.EventProducesEntity],
    entity_points_to: List[schemas.EntityPointsToEvent],
    causal_links: List[schemas.CausalLink],
    # --- NEW ARGUMENTS ---
    graph_model: str = "chain",
    semantic_links: Optional[List[schemas.SemanticLink]] = None,
    scenes: Optional[List[schemas.Scene]] = None
) -> Tuple[List[schemas.GenericNode], List[schemas.GenericRelationship]]:
    """
    Maps the specific pipeline data into generic lists of nodes and relationships.
    This is where all the domain-specific logic now lives.
    """
    nodes: Dict[str, schemas.GenericNode] = {}
    relationships: List[schemas.GenericRelationship] = []
    
    if semantic_links is None:
        semantic_links = []
    if scenes is None:
        scenes = []

    # 1. Map Events to Nodes (Same for both models)
    for ev in events:
        nodes[ev.id] = schemas.GenericNode(
            uid=ev.id,
            label="Event",
            properties=_escape_props({
                "id": ev.id,
                "name": ev.name,
                "eventType": ev.eventType,
                "actionType": ev.actionType,
                "source_quote": utils._truncate_safe(ev.source_quote, 300),
                "causeWeight": ev.causeWeight or 0.0,
                "confidence": ev.confidence,
                "sequence": ev.sequence,
                "chapter": ev.chapter,
                "time": ev.time or "",
                "location": ev.location or ""
            })
        )

    # 2. Map Entities to Nodes (Uses CANONICAL IDs for 'star' model)
    entities_by_type = defaultdict(dict)
    for prod in event_produces:
        entities_by_type[prod.entity_type][prod.entity_id] = prod.entity_name

    # Agent nodes
    all_agents = {}
    all_agents.update(entities_by_type.get("actor", {}))
    all_agents.update(entities_by_type.get("patient", {}))
    for agent_id, agent_name in all_agents.items():
        # --- MODIFIED FOR 'star' MODEL ---
        # Use helper to strip quotes
        safe_name = _sanitize_name_for_id(agent_name)
        canonical_id = f"agent_{safe_name}"
        
        # Use canonical_id for 'star', but original entity_id for 'chain'
        node_uid = canonical_id if graph_model == "star" else agent_id
        
        if node_uid not in nodes: # Avoid duplicates
            nodes[node_uid] = schemas.GenericNode(
                uid=node_uid,
                label="Agent",
                properties=_escape_props({"id": node_uid, "name": agent_name})
            )
        # --- END MODIFICATION ---

    # WhyFactor nodes
    for wf_id, wf_name in entities_by_type.get("whyfactor", {}).items():
        # --- MODIFIED FOR 'star' MODEL ---
        safe_name = _sanitize_name_for_id(wf_name)[:30] # Truncate for ID
        canonical_id = f"whyfactor_{safe_name}"
        node_uid = canonical_id if graph_model == "star" else wf_id
        
        if node_uid not in nodes:
            nodes[node_uid] = schemas.GenericNode(
                uid=node_uid,
                label="WhyFactor",
                properties=_escape_props({"id": node_uid, "factor": wf_name})
            )
        # --- END MODIFICATION ---

    # Place nodes
    for place_id, place_name in entities_by_type.get("place", {}).items():
        # --- MODIFIED FOR 'star' MODEL ---
        safe_name = _sanitize_name_for_id(place_name)
        canonical_id = f"place_{safe_name}"
        node_uid = canonical_id if graph_model == "star" else place_id
        
        if node_uid not in nodes:
            nodes[node_uid] = schemas.GenericNode(
                uid=node_uid,
                label="Place",
                properties=_escape_props({"id": node_uid, "name": place_name})
            )
        # --- END MODIFICATION ---


    # --- NEW: Map Scene Nodes ---
    for scene in scenes:
        if scene.id not in nodes:
            nodes[scene.id] = schemas.GenericNode(
                uid=scene.id,
                label="Scene",
                properties=_escape_props({
                    "id": scene.id,
                    "theme": scene.theme,
                    "chapter": scene.chapter,
                    "confidence": scene.confidence
                })
            )
        # Create relationships from Scene to Event
        for event_id in scene.event_ids:
            if event_id in nodes: # Only link if event exists
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=scene.id,
                    end_node_uid=event_id,
                    rel_type="INCLUDES",
                    properties={}
                ))
    # --- END NEW ---

    # --- Handle Graph Model (Feature 2) ---
    if graph_model == "chain":
        print("[graph_mapper] Using 'chain' model (Event->Entity->Event)")
        # 3. Map Event-Entity Production Relationships
        for prod in event_produces:
            prop_key = "strength"
            if prod.entity_type == "whyfactor": prop_key = "weight"
            elif prod.entity_type == "place": prop_key = "specificity"
            
            relationships.append(schemas.GenericRelationship(
                start_node_uid=prod.event_id,
                end_node_uid=prod.entity_id,
                rel_type=prod.relationship,
                properties={prop_key: prod.strength}
            ))

        # 4. Map Entity-Event Pointing Relationships
        for ept in entity_points_to:
            prop_key = "strength"
            if ept.entity_type == "whyfactor": prop_key = "weight"
            elif ept.entity_type == "place": prop_key = "specificity"

            relationships.append(schemas.GenericRelationship(
                start_node_uid=ept.entity_id,
                end_node_uid=ept.next_event_id,
                rel_type=ept.relationship,
                properties={prop_key: ept.strength}
            ))
    
    elif graph_model == "star":
        print("[graph_mapper] Using 'star' model (Canonical Entity -> Events)")
        # 3. Map Canonical Entities directly to Events
        for prod in event_produces:
            canonical_id = None
            rel_type = None
            prop_key = "strength"

            safe_name = _sanitize_name_for_id(prod.entity_name)

            if prod.entity_type == "actor":
                canonical_id = f"agent_{safe_name}"
                rel_type = "ACTS_IN"
            elif prod.entity_type == "patient":
                canonical_id = f"agent_{safe_name}"
                rel_type = "AFFECTED_IN"
            elif prod.entity_type == "place":
                canonical_id = f"place_{safe_name}"
                rel_type = "HOSTS"
                prop_key = "specificity"
            elif prod.entity_type == "whyfactor":
                canonical_id = f"whyfactor_{safe_name[:30]}"
                rel_type = "MOTIVATES"
                prop_key = "weight"
            
            if canonical_id and rel_type:
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=canonical_id, # <-- Canonical entity
                    end_node_uid=prod.event_id,    # <-- Specific event
                    rel_type=rel_type,
                    properties={prop_key: prod.strength}
                ))

    # 5. Map Event-Event (Follows) Relationships (Same for both models)
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
    
    # 6. Map Event-Event (Causal) Relationships (Same for both models)
    for link in causal_links:
        relationships.append(schemas.GenericRelationship(
            start_node_uid=link.cause_id,
            end_node_uid=link.effect_id,
            rel_type="CAUSES", # You could also use link.relationType
            properties=_escape_props({
                "relationType": link.relationType,
                "mechanism": utils._truncate_safe(link.mechanism, 200),
                "sign": link.sign,
                "weight": link.weight,
                "confidence": link.confidence,
                "cause_seq": link.cause_sequence,
                "effect_seq": link.effect_sequence
            })
        ))

    # --- NEW: Map Semantic Links ---
    for link in semantic_links:
        for source_id in link.source_event_ids:
            for target_id in link.target_event_ids:
                if source_id in nodes and target_id in nodes:
                    relationships.append(schemas.GenericRelationship(
                        start_node_uid=source_id,
                        end_node_uid=target_id,
                        rel_type=link.relation.upper(), # "EXPLANATION", etc.
                        properties=_escape_props({
                            "cue": ", ".join(link.cue) if link.cue else "",
                            "confidence": link.confidence
                        })
                    ))
    # --- END NEW ---

    return list(nodes.values()), relationships