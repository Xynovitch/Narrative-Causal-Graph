from collections import defaultdict
from typing import List, Dict, Tuple
from .schemas import CEKEvent, EventProducesEntity, EntityPointsToEvent
from .utils import _make_id

def propagate_context_attributes(
    events: List[CEKEvent]
) -> List[CEKEvent]:
    """
    Pass 1: Iterate through events and propagate Time/Location attributes.
    If an event has no location/time, it inherits from the previous one.
    """
    print("[context] Propagating Location and Time attributes within the event chain...")
    
    current_location = None
    current_time = None
    
    # Assumes events are already sorted by sequence
    # We sort just in case to ensure linear narrative flow
    sorted_events = sorted(events, key=lambda x: x.sequence)
    
    for event in sorted_events:
        # 1. Location Propagation
        if event.location_context:
            # If explicit location exists, update current context
            current_location = event.location_context
        elif current_location:
            # If no explicit location, inherit from current context
            event.location_context = current_location
            
        # 2. Time Propagation
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
    Pass 2: Propagate *Entity* context (Actors, WhyFactors).
    Location/Time are now handled in Pass 1 (attributes), so we skip them here
    to avoid creating duplicate nodes.
    """
    print("[context] Propagating stateful entity context (Actors, WhyFactors)...")
    
    # {actor_key: (actor_name, actor_type, relationship)}
    current_actors = {}
    # [(factor_name, factor_type, relationship, strength), ...]
    current_whyfactors = []

    newly_produced = [] 
    
    # Build lookup
    prods_by_event = defaultdict(list)
    for prod in event_produces:
        prods_by_event[prod.event_id].append(prod)

    # Sort events
    sorted_events = sorted(events, key=lambda x: x.sequence)

    for event in sorted_events:
        explicit_actors = {}
        explicit_whyfactors = []
        
        # Check what this event explicitly provides
        for prod in prods_by_event[event.id]:
            if prod.entity_type in ['actor', 'patient']:
                key = prod.entity_name.lower()
                explicit_actors[key] = (prod.entity_name, prod.entity_type, prod.relationship)
            elif prod.entity_type == 'whyfactor':
                explicit_whyfactors.append(prod)
            # NOTE: We ignore 'place' here because it is now an attribute.

        # 1. Propagate Actors
        if explicit_actors:
            current_actors = explicit_actors
        elif current_actors:
            # Inherit previous actors
            for actor_key, (actor_name, actor_type, actor_rel) in current_actors.items():
                new_actor_id = _make_id(f"agent_{actor_key.replace(' ', '_')}")
                
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=new_actor_id,
                    entity_name=actor_name,
                    entity_type=actor_type,
                    relationship=actor_rel,
                    strength=0.5 # Lower strength for inferred
                )
                newly_produced.append(new_prod)
                entity_occurrences[f"{actor_type}:{actor_key}"].append((event.id, event.sequence))

        # 2. Propagate WhyFactors
        if explicit_whyfactors:
            current_whyfactors = explicit_whyfactors
        elif current_whyfactors:
            # Inherit previous whyfactors
            for w_prod in current_whyfactors:
                # Re-create ID logic or reuse ID
                factor_key = w_prod.entity_name.lower()
                
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=w_prod.entity_id,
                    entity_name=w_prod.entity_name,
                    entity_type=w_prod.entity_type,
                    relationship=w_prod.relationship,
                    strength=w_prod.strength
                )
                newly_produced.append(new_prod)
                entity_occurrences[f"whyfactor:{factor_key}"].append((event.id, event.sequence))
    
    # Re-sort occurrence lists
    for key in entity_occurrences:
        entity_occurrences[key].sort(key=lambda x: x[1])
        
    return newly_produced, entity_occurrences

def create_entity_to_event_links(
    entity_occurrences: Dict[str, List[Tuple[str, int]]],
    event_produces: List[EventProducesEntity]
) -> List[EntityPointsToEvent]:
    """Create Entity -> NextEvent links."""
    print("[linking] Creating entity -> event links...")
    new_links = []
    
    for entity_key, occurrences in entity_occurrences.items():
        if "place:" in entity_key: 
            continue # Skip places (handled as attributes)
            
        try:
            entity_type, entity_name = entity_key.split(":", 1)
        except ValueError:
            continue
            
        # Create chain links
        for i in range(len(occurrences) - 1):
            curr_evt, _ = occurrences[i]
            next_evt, _ = occurrences[i+1]
            
            # Resolve ID and Strength by looking up the production
            match_prod = next(
                (p for p in event_produces if p.event_id == curr_evt and p.entity_name.lower() == entity_name.lower()),
                None
            )
            
            if match_prod:
                # Map entity type to relationship
                rel_map = {
                    "actor": "ACTS_IN",
                    "patient": "AFFECTED_IN",
                    "whyfactor": "MOTIVATES"
                }
                
                if entity_type in rel_map:
                    new_links.append(EntityPointsToEvent(
                        entity_id=match_prod.entity_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        next_event_id=next_evt,
                        relationship=rel_map[entity_type],
                        strength=match_prod.strength
                    ))
                    
    return new_links