from collections import defaultdict
from typing import List, Dict, Any, Tuple
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

def map_to_generic_graph(
    events: List[schemas.CEKEvent],
    event_produces: List[schemas.EventProducesEntity],
    entity_points_to: List[schemas.EntityPointsToEvent],
    causal_links: List[schemas.CausalLink]
) -> Tuple[List[schemas.GenericNode], List[schemas.GenericRelationship]]:
    """
    Maps the specific pipeline data into generic lists of nodes and relationships.
    This is where all the domain-specific logic now lives.
    """
    nodes: Dict[str, schemas.GenericNode] = {}
    relationships: List[schemas.GenericRelationship] = []

    # 1. Map Events to Nodes
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

    # 2. Map Entities (from event_produces) to Nodes
    entities_by_type = defaultdict(dict)
    for prod in event_produces:
        entities_by_type[prod.entity_type][prod.entity_id] = prod.entity_name

    # Agent nodes
    all_agents = {}
    all_agents.update(entities_by_type.get("actor", {}))
    all_agents.update(entities_by_type.get("patient", {}))
    for agent_id, agent_name in all_agents.items():
        if agent_id not in nodes: # Avoid duplicates
            nodes[agent_id] = schemas.GenericNode(
                uid=agent_id,
                label="Agent",
                properties=_escape_props({"id": agent_id, "name": agent_name})
            )

    # WhyFactor nodes
    for wf_id, wf_name in entities_by_type.get("whyfactor", {}).items():
        if wf_id not in nodes:
            nodes[wf_id] = schemas.GenericNode(
                uid=wf_id,
                label="WhyFactor",
                properties=_escape_props({"id": wf_id, "factor": wf_name})
            )

    # Place nodes
    for place_id, place_name in entities_by_type.get("place", {}).items():
        if place_id not in nodes:
            nodes[place_id] = schemas.GenericNode(
                uid=place_id,
                label="Place",
                properties=_escape_props({"id": place_id, "name": place_name})
            )

    # 3. Map Event-Entity Production Relationships
    for prod in event_produces:
        # Define property key based on type
        prop_key = "strength"
        if prod.entity_type == "whyfactor":
            prop_key = "weight"
        elif prod.entity_type == "place":
            prop_key = "specificity"
        
        relationships.append(schemas.GenericRelationship(
            start_node_uid=prod.event_id,
            end_node_uid=prod.entity_id,
            rel_type=prod.relationship,
            properties={prop_key: prod.strength}
        ))

    # 4. Map Entity-Event Pointing Relationships
    for ept in entity_points_to:
        prop_key = "strength"
        if ept.entity_type == "whyfactor":
            prop_key = "weight"
        elif ept.entity_type == "place":
            prop_key = "specificity"

        relationships.append(schemas.GenericRelationship(
            start_node_uid=ept.entity_id,
            end_node_uid=ept.next_event_id,
            rel_type=ept.relationship,
            properties={prop_key: ept.strength}
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
    
    # 6. Map Event-Event (Causal) Relationships
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

    return list(nodes.values()), relationships