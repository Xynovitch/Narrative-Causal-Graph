"""
Chain Model Archive

Standalone script implementing the chain-model graph export pattern.
NOT used by the main pipeline (which uses the star model exclusively).

Chain model: Event -[:PRODUCES]-> EntityInstance -[:ACTS_IN]-> NextEvent
Each entity is scoped to the event that produced it, forming a chain.

Usage (standalone, post-pipeline):
    from chain_model import build_chain_graph
    nodes, relationships = build_chain_graph(all_events, all_produces, causal_links)
"""

from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

# These imports assume this script is run from the project root
from cekg_pipeline import schemas, utils


def _sanitize_name_for_id(name: str) -> str:
    if not name:
        return "unknown"
    clean = name.lower().replace(' ', '_')
    clean = clean.replace('"', '').replace("'", "").replace(":", "").replace("\\", "")
    return clean


def _escape_props(props: Dict[str, Any]) -> Dict[str, Any]:
    escaped = {}
    for k, v in props.items():
        if isinstance(v, str):
            escaped[k] = utils._escape_cypher_string(v)
        else:
            escaped[k] = v
    return escaped


def build_chain_entity_links(
    event_produces: List[schemas.EventProducesEntity],
    entity_occurrences: Dict[str, List[Tuple[str, int]]]
) -> List[Tuple[str, str, str, float]]:
    """
    Build chain-model entity->event edges.
    Each entity instance (scoped to an event) points to the NEXT event that entity appears in.

    Returns a list of (entity_id, next_event_id, relationship, strength) tuples.
    """
    # Build lookup: (event_id, entity_name_lower, entity_type) -> (entity_id, strength)
    prod_lookup = {}
    for p in event_produces:
        key = (p.event_id, p.entity_name.strip().lower(), p.entity_type)
        prod_lookup[key] = (p.entity_id, p.strength)

    rel_type_map = {"actor": "ACTS_IN", "patient": "AFFECTED_IN", "whyfactor": "MOTIVATES"}
    links = []

    for entity_key, occurrences in entity_occurrences.items():
        if "place:" in entity_key:
            continue
        try:
            entity_type, entity_name = entity_key.split(":", 1)
        except ValueError:
            continue

        rel = rel_type_map.get(entity_type)
        if not rel:
            continue

        clean_name = entity_name.strip().lower()

        for i in range(len(occurrences) - 1):
            curr_evt, _ = occurrences[i]
            next_evt, _ = occurrences[i + 1]
            found = prod_lookup.get((curr_evt, clean_name, entity_type))
            if found:
                ent_id, strength = found
                links.append((ent_id, next_evt, rel, strength))

    return links


def build_chain_graph(
    events: List[schemas.CEKEvent],
    event_produces: List[schemas.EventProducesEntity],
    causal_links: List[schemas.CausalLink],
    entity_occurrences: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    semantic_links: Optional[List[schemas.SemanticLink]] = None,
    scenes: Optional[List[schemas.Scene]] = None
) -> Tuple[List[schemas.GenericNode], List[schemas.GenericRelationship]]:
    """
    Build chain-model graph: Event -> PRODUCES -> EntityInstance -> ACTS_IN -> NextEvent.
    Entity nodes are scoped per-event (not canonical).
    """
    nodes: Dict[str, schemas.GenericNode] = {}
    relationships: List[schemas.GenericRelationship] = []

    if semantic_links is None:
        semantic_links = []
    if scenes is None:
        scenes = []
    if entity_occurrences is None:
        entity_occurrences = {}

    # 1. Event nodes
    for ev in events:
        nodes[ev.id] = schemas.GenericNode(
            uid=ev.id,
            label="Event",
            properties=_escape_props({
                "id": ev.id,
                "name": ev.raw_description,
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

    # 2. Entity nodes (event-scoped IDs)
    for prod in event_produces:
        if prod.entity_id not in nodes:
            label = "Agent" if prod.entity_type in ['actor', 'patient'] else "WhyFactor"
            nodes[prod.entity_id] = schemas.GenericNode(
                uid=prod.entity_id,
                label=label,
                properties=_escape_props({
                    "id": prod.entity_id,
                    "name": prod.entity_name,
                    "entityType": prod.entity_type,
                    "agentType": prod.agent_type or "STRUCTURAL_AGENT",
                    "theory": prod.theory or "@McKee"
                })
            )

    # 3. Scene nodes
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
                    "time": scene.time_period or ""
                })
            )
        for event_id in scene.included_event_ids:
            if event_id in nodes:
                relationships.append(schemas.GenericRelationship(
                    start_node_uid=scene.id,
                    end_node_uid=event_id,
                    rel_type="INCLUDES",
                    properties={}
                ))

    # 4a. Event -> PRODUCES -> EntityInstance
    for prod in event_produces:
        relationships.append(schemas.GenericRelationship(
            start_node_uid=prod.event_id,
            end_node_uid=prod.entity_id,
            rel_type=prod.relationship,
            properties={"strength": prod.strength}
        ))

    # 4b. EntityInstance -> ACTS_IN -> NextEvent
    for ent_id, next_evt_id, rel, strength in build_chain_entity_links(event_produces, entity_occurrences):
        relationships.append(schemas.GenericRelationship(
            start_node_uid=ent_id,
            end_node_uid=next_evt_id,
            rel_type=rel,
            properties={"strength": strength}
        ))

    # 5. FOLLOWS edges
    for i in range(len(events) - 1):
        ev1, ev2 = events[i], events[i + 1]
        if ev1.chapter == ev2.chapter:
            relationships.append(schemas.GenericRelationship(
                start_node_uid=ev1.id,
                end_node_uid=ev2.id,
                rel_type="FOLLOWS",
                properties={}
            ))

    # 6. Causal links
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

    # 7. Semantic links
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

    print(f"[chain_model] Built {len(nodes)} nodes and {len(relationships)} relationships")
    return list(nodes.values()), relationships
