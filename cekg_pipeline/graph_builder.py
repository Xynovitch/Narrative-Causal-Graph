from collections import defaultdict
from typing import List, Dict, Tuple
from .schemas import CEKEvent, EventProducesEntity
from .utils import _make_id

def _generate_entity_id(name: str, type_prefix: str) -> str:
    """Generates a canonical global Entity ID (e.g., "agent_pip")."""
    clean_name = name.strip().lower().replace(' ', '_')
    base_id = f"{type_prefix}_{clean_name}"
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
    entity_occurrences: Dict[str, List[Tuple[str, int]]]
) -> Tuple[List[EventProducesEntity], Dict[str, List[Tuple[str, int]]]]:
    """
    Pass 2: Propagate Entity context (Actors, WhyFactors) across events.
    When an entity is not explicitly mentioned in an event, it inherits the
    currently active entities from the narrative context.

    This implements "contextual persistence" - entities remain active until
    new entities are introduced.
    """
    print("[context] Propagating entity context...")
    
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
                new_actor_id = _generate_entity_id(actor_name, "agent")
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
                new_factor_id = _generate_entity_id(w_prod.entity_name, "why")
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

