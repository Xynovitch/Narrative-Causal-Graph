from collections import defaultdict
from typing import List, Dict, Tuple
from .schemas import CEKEvent, EventProducesEntity, EntityPointsToEvent
from .utils import _make_id

def propagate_context(
    events: List[CEKEvent], 
    event_produces: List[EventProducesEntity],
    entity_occurrences: Dict[str, List[Tuple[str, int]]]
) -> Tuple[List[EventProducesEntity], Dict[str, List[Tuple[str, int]]]]:
    """
    Pass 2: Iterate through all events and propagate location, actor,
    and whyfactor context to fill in the blanks.
    
    NOTE: This function MUTATES the `events` list in place (to update
    location) but returns NEW lists for `EventProducesEntity` and
    `entity_occurrences`.
    """
    print("[context] Propagating stateful context (locations, actors, whyfactors)...")
    current_location_name = None
    # {actor_key: (actor_name, actor_type, relationship)}
    current_actors = {}
    # [(factor_name, factor_type, relationship, strength), ...]
    current_whyfactors = []

    newly_produced = [] # To store new links we create
    
    # Build a lookup for explicitly produced entities
    prods_by_event = defaultdict(list)
    for prod in event_produces:
        prods_by_event[prod.event_id].append(prod)

    # IMPORTANT: Assumes `events` is already sorted by sequence
    for event in events:
        
        explicit_actors = {}
        explicit_location_name = None
        explicit_whyfactors = []
        
        # Check for entities explicitly extracted for this event
        for prod in prods_by_event[event.id]:
            if prod.entity_type == 'actor' or prod.entity_type == 'patient':
                key = prod.entity_name.lower()
                explicit_actors[key] = (prod.entity_name, prod.entity_type, prod.relationship)
            elif prod.entity_type == 'place':
                explicit_location_name = prod.entity_name
            elif prod.entity_type == 'whyfactor':
                explicit_whyfactors.append((prod.entity_name, 'whyfactor', 'PRODUCES_MOTIVATION', prod.strength))

        
        # 1. Propagate Location
        if explicit_location_name:
            # This event sets a new location
            current_location_name = explicit_location_name
        elif current_location_name:
            # This event is missing a location. Assign the current one.
            event.location = current_location_name # Update event object
            loc_key = current_location_name.lower()
            new_loc_id = _make_id(f"place_{loc_key.replace(' ', '_')}")
            event.location_id = new_loc_id # Update event object
            
            new_prod = EventProducesEntity(
                event_id=event.id,
                entity_id=new_loc_id,
                entity_name=current_location_name,
                entity_type="place",
                relationship="PRODUCES_LOCATION",
                strength=0.5 # Lower strength for inferred context
            )
            newly_produced.append(new_prod)
            entity_occurrences[f"place:{loc_key}"].append((event.id, event.sequence))

        # 2. Propagate Actors
        if explicit_actors:
            # This event defines a new set of actors
            current_actors = explicit_actors
        elif current_actors:
            # This event is missing actors. Assign the current ones.
            for actor_key, (actor_name, actor_type, actor_rel) in current_actors.items():
                new_actor_id = _make_id(f"agent_{actor_key.replace(' ', '_')}")
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=new_actor_id,
                    entity_name=actor_name,
                    entity_type=actor_type,
                    relationship=actor_rel,
                    strength=0.5 # Lower strength for inferred context
                )
                newly_produced.append(new_prod)
                entity_occurrences[f"{actor_type}:{actor_key}"].append((event.id, event.sequence))

        # 3. Propagate WhyFactors
        if explicit_whyfactors:
            # This event defines a new set of whyfactors
            current_whyfactors = explicit_whyfactors
        elif current_whyfactors:
            # This event is missing whyfactors. Assign the current ones.
            for factor_name, factor_type, factor_rel, factor_strength in current_whyfactors:
                factor_key = factor_name.lower()
                new_factor_id = _make_id(f"why_{factor_key[:30].replace(' ', '_')}")
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=new_factor_id,
                    entity_name=factor_name,
                    entity_type=factor_type,
                    relationship=factor_rel,
                    strength=factor_strength # Propagate original strength
                )
                newly_produced.append(new_prod)
                entity_occurrences[f"whyfactor:{factor_key}"].append((event.id, event.sequence))
    
    print(f"[context] Propagated context, created {len(newly_produced)} new entity links.")
    
    # Re-sort all occurrence lists since we added new ones in parallel
    for key in entity_occurrences:
        entity_occurrences[key].sort(key=lambda x: x[1])
        
    return newly_produced, entity_occurrences

def create_entity_to_event_links(
    entity_occurrences: Dict[str, List[Tuple[str, int]]],
    event_produces: List[EventProducesEntity]
) -> List[EntityPointsToEvent]:
    """Create Entity -[:X]-> NextEvent links based on entity occurrences"""
    print("[linking] Creating entity→event links...")
    
    new_links = []
    
    for entity_key, occurrences in entity_occurrences.items():
        # Already sorted by propagate_context
        
        # Parse entity type and name
        try:
            entity_type, entity_name = entity_key.split(":", 1)
            target_name_lower = entity_name.strip().lower()
        except ValueError:
            print(f"[linking] Skipping malformed entity key: {entity_key}")
            continue
        
        # Create links from each occurrence to the next
        for i in range(len(occurrences) - 1):
            current_event_id, current_seq = occurrences[i]
            next_event_id, next_seq = occurrences[i + 1]
            
            entity_id = None
            strength = 0.5  # Default strength
            
            # Find the entity_id and strength produced by the current event
            for prod in event_produces:
                if not prod.entity_name:
                    continue
                
                prod_name = prod.entity_name.strip().lower()
                
                # Match on event, type, and name
                if prod.event_id == current_event_id and \
                   prod.entity_type == entity_type and \
                   prod_name == target_name_lower:
                    
                    entity_id = prod.entity_id
                    strength = prod.strength  # <-- CAPTURE STRENGTH HERE
                    break # Found the match
            
            if entity_id:
                # Determine relationship type based on the entity type
                if entity_type == "actor":
                    rel_type = "ACTS_IN"
                elif entity_type == "patient":
                    rel_type = "AFFECTED_IN"
                elif entity_type == "whyfactor":
                    rel_type = "MOTIVATES"
                elif entity_type == "place":
                    rel_type = "HOSTS"
                else:
                    continue
                
                # Add the new Entity->Event link
                new_links.append(EntityPointsToEvent(
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    next_event_id=next_event_id,
                    relationship=rel_type,
                    strength=strength
                ))
    
    print(f"[linking] Created {len(new_links)} entity→event links")
    return new_links