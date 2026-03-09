from collections import defaultdict
from typing import List, Dict, Tuple
from .schemas import CEKEvent, EventProducesEntity, EntityPointsToEvent
from .utils import _make_id

def _generate_entity_id(name: str, type_prefix: str, event_id: str, graph_model: str) -> str:
    """
    Generates an Entity ID based on the graph model.
    - Star model: Canonical global ID (e.g., "agent_pip")
    - Chain model: Event-specific instance ID (e.g., "agent_pip_event_001")
    """
    clean_name = name.strip().lower().replace(' ', '_')
    base_id = f"{type_prefix}_{clean_name}"
    
    if graph_model == "chain":
        return _make_id(f"{base_id}_{event_id}")
    else:
        return _make_id(base_id)

def propagate_context_attributes(events: List[CEKEvent]) -> List[CEKEvent]:
    """
    Pass 1: Propagate Location and Time attributes within the event chain.
    This ensures that events inherit context from previous events when not explicitly stated.
    """
    print("[context] Propagating Location and Time attributes within the event chain...")
    
    current_location = None
    current_time = None
    
    sorted_events = sorted(events, key=lambda x: x.sequence)
    
    for event in sorted_events:
        # Propagate location
        if event.location_context:
            current_location = event.location_context
        elif current_location:
            event.location_context = current_location
            
        # Propagate time
        if event.time_context:
            current_time = event.time_context
        elif current_time:
            event.time_context = current_time

    return sorted_events

def propagate_context(
    events: List[CEKEvent], 
    event_produces: List[EventProducesEntity],
    entity_occurrences: Dict[str, List[Tuple[str, int]]],
    graph_model: str = "star"
) -> Tuple[List[EventProducesEntity], Dict[str, List[Tuple[str, int]]]]:
    """
    Pass 2: Propagate Entity context (Actors, WhyFactors) across events.
    When an entity is not explicitly mentioned in an event, it inherits the
    currently active entities from the narrative context.
    
    This implements "contextual persistence" - entities remain active until
    new entities are introduced.
    """
    print(f"[context] Propagating entity context (Mode: {graph_model})...")
    
    current_actors = {}  # {actor_key: (actor_name, actor_type, relationship)}
    current_whyfactors = []  # [(factor_name, factor_type, relationship, strength)]
    newly_produced = []
    
    # Build lookup for explicitly produced entities
    prods_by_event = defaultdict(list)
    for prod in event_produces:
        prods_by_event[prod.event_id].append(prod)

    sorted_events = sorted(events, key=lambda x: x.sequence)

    for event in sorted_events:
        explicit_actors = {}
        explicit_whyfactors = []
        
        # Identify explicitly mentioned entities in this event
        for prod in prods_by_event[event.id]:
            if prod.entity_type in ['actor', 'patient']:
                key = prod.entity_name.lower()
                explicit_actors[key] = (prod.entity_name, prod.entity_type, prod.relationship)
            elif prod.entity_type == 'whyfactor':
                explicit_whyfactors.append(prod)

        # Propagate Actors
        if explicit_actors:
            # New actors introduced - update context
            current_actors = explicit_actors
        elif current_actors:
            # No new actors - propagate existing context
            for actor_key, (actor_name, actor_type, actor_rel) in current_actors.items():
                new_actor_id = _generate_entity_id(actor_name, "agent", event.id, graph_model)
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=new_actor_id,
                    entity_name=actor_name,
                    entity_type=actor_type,
                    relationship=actor_rel,
                    strength=0.5  # Lower strength for inferred context
                )
                newly_produced.append(new_prod)
                entity_occurrences[f"{actor_type}:{actor_key}"].append((event.id, event.sequence))

        # Propagate WhyFactors
        if explicit_whyfactors:
            # New motivations introduced - update context
            current_whyfactors = explicit_whyfactors
        elif current_whyfactors:
            # No new motivations - propagate existing context
            for w_prod in current_whyfactors:
                new_factor_id = _generate_entity_id(w_prod.entity_name, "why", event.id, graph_model)
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=new_factor_id,
                    entity_name=w_prod.entity_name,
                    entity_type=w_prod.entity_type,
                    relationship=w_prod.relationship,
                    strength=w_prod.strength
                )
                newly_produced.append(new_prod)
                entity_occurrences[f"whyfactor:{w_prod.entity_name.lower()}"].append((event.id, event.sequence))
    
    # Sort all occurrence lists by sequence
    for key in entity_occurrences:
        entity_occurrences[key].sort(key=lambda x: x[1])
        
    print(f"[context] Created {len(newly_produced)} propagated entity links")
    return newly_produced, entity_occurrences

def create_entity_to_event_links(
    entity_occurrences: Dict[str, List[Tuple[str, int]]],
    event_produces: List[EventProducesEntity],
    graph_model: str = "star"
) -> List[EntityPointsToEvent]:
    """
    Create Entity -> Event links with O(1) lookup optimization.
    
    Star Mode: Canonical Entity -> ACTS_IN -> Event (entity points to its own event)
    Chain Mode: Entity Instance -> ACTS_IN -> NextEvent (entity points to next event)
    """
    print(f"[linking] Creating entity -> event links (Mode: {graph_model})...")
    new_links = []
    
    # Build O(1) lookup map: (event_id, entity_name_lower, entity_type) -> (entity_id, strength)
    prod_lookup = {}
    for p in event_produces:
        key = (p.event_id, p.entity_name.strip().lower(), p.entity_type)
        prod_lookup[key] = (p.entity_id, p.strength)

    for entity_key, occurrences in entity_occurrences.items():
        # Skip place entities (handled separately if needed)
        if "place:" in entity_key:
            continue
        
        try:
            entity_type, entity_name = entity_key.split(":", 1)
            clean_name = entity_name.strip().lower()
        except ValueError:
            continue
            
        # Determine relationship type
        if entity_type == "actor":
            rel = "ACTS_IN"
        elif entity_type == "patient":
            rel = "AFFECTED_IN"
        elif entity_type == "whyfactor":
            rel = "MOTIVATES"
        else:
            continue

        if graph_model == "star":
            # STAR: Entity points to its OWN event (canonical entity -> all its events)
            for evt_id, seq in occurrences:
                found = prod_lookup.get((evt_id, clean_name, entity_type))
                if found:
                    ent_id, strength = found
                    new_links.append(EntityPointsToEvent(
                        entity_id=ent_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        next_event_id=evt_id,
                        relationship=rel,
                        strength=strength
                    ))
        else:
            # CHAIN: Entity instance points to NEXT event
            for i in range(len(occurrences) - 1):
                curr_evt, _ = occurrences[i]
                next_evt, _ = occurrences[i + 1]
                
                # Lookup entity from CURRENT event
                found = prod_lookup.get((curr_evt, clean_name, entity_type))
                if found:
                    ent_id, strength = found
                    new_links.append(EntityPointsToEvent(
                        entity_id=ent_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        next_event_id=next_evt,
                        relationship=rel,
                        strength=strength
                    ))
                    
    print(f"[linking] Created {len(new_links)} entity -> event links")
    return new_links