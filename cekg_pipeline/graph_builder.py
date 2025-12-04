from collections import defaultdict
from typing import List, Dict, Tuple
from .schemas import CEKEvent, EventProducesEntity, EntityPointsToEvent
from .utils import _make_id

def _generate_entity_id(name: str, type_prefix: str, event_id: str, graph_model: str) -> str:
    """
    Generates an Entity ID based on the graph model.
    """
    clean_name = name.strip().lower().replace(' ', '_')
    base_id = f"{type_prefix}_{clean_name}"
    
    if graph_model == "chain":
        # Create a unique instance for this specific event
        return _make_id(f"{base_id}_{event_id}")
    else:
        # Default/Star: Canonical global ID
        return _make_id(base_id)

def propagate_context_attributes(
    events: List[CEKEvent]
) -> List[CEKEvent]:
    """Pass 1: Propagate Location and Time attributes."""
    print("[context] Propagating Location and Time attributes within the event chain...")
    
    current_location = None
    current_time = None
    
    sorted_events = sorted(events, key=lambda x: x.sequence)
    
    for event in sorted_events:
        if event.location_context:
            current_location = event.location_context
        elif current_location:
            event.location_context = current_location
            
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
    """Pass 2: Propagate Entity context (Actors, WhyFactors)."""
    print(f"[context] Propagating entity context (Mode: {graph_model})...")
    
    current_actors = {}
    current_whyfactors = []
    newly_produced = [] 
    
    prods_by_event = defaultdict(list)
    for prod in event_produces:
        prods_by_event[prod.event_id].append(prod)

    sorted_events = sorted(events, key=lambda x: x.sequence)

    for event in sorted_events:
        explicit_actors = {}
        explicit_whyfactors = []
        
        for prod in prods_by_event[event.id]:
            if prod.entity_type in ['actor', 'patient']:
                key = prod.entity_name.lower()
                explicit_actors[key] = (prod.entity_name, prod.entity_type, prod.relationship)
            elif prod.entity_type == 'whyfactor':
                explicit_whyfactors.append(prod)

        # Propagate Actors
        if explicit_actors:
            current_actors = explicit_actors
        elif current_actors:
            for actor_key, (actor_name, actor_type, actor_rel) in current_actors.items():
                new_actor_id = _generate_entity_id(actor_name, "agent", event.id, graph_model)
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=new_actor_id,
                    entity_name=actor_name,
                    entity_type=actor_type,
                    relationship=actor_rel,
                    strength=0.5
                )
                newly_produced.append(new_prod)
                entity_occurrences[f"{actor_type}:{actor_key}"].append((event.id, event.sequence))

        # Propagate WhyFactors
        if explicit_whyfactors:
            current_whyfactors = explicit_whyfactors
        elif current_whyfactors:
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
    
    for key in entity_occurrences:
        entity_occurrences[key].sort(key=lambda x: x[1])
        
    return newly_produced, entity_occurrences

def create_entity_to_event_links(
    entity_occurrences: Dict[str, List[Tuple[str, int]]],
    event_produces: List[EventProducesEntity],
    graph_model: str = "star"
) -> List[EntityPointsToEvent]:
    """
    Create Entity -> Event links. Optimized with O(1) lookup.
    """
    print(f"[linking] Creating entity -> event links (Mode: {graph_model})...")
    new_links = []
    
    # OPTIMIZATION: Build a lookup map to avoid O(N^2) loop
    # Key: (event_id, entity_name_lower, entity_type) -> (entity_id, strength)
    prod_lookup = {}
    for p in event_produces:
        key = (p.event_id, p.entity_name.strip().lower(), p.entity_type)
        prod_lookup[key] = (p.entity_id, p.strength)

    for entity_key, occurrences in entity_occurrences.items():
        if "place:" in entity_key: continue
        
        try:
            entity_type, entity_name = entity_key.split(":", 1)
            clean_name = entity_name.strip().lower()
        except ValueError: continue
            
        # Determine Relationship Type
        if entity_type == "actor": rel = "ACTS_IN"
        elif entity_type == "patient": rel = "AFFECTED_IN"
        elif entity_type == "whyfactor": rel = "MOTIVATES"
        else: continue

        if graph_model == "star":
            # STAR: Link entity to its OWN event
            for evt_id, seq in occurrences:
                # Lookup O(1)
                found = prod_lookup.get((evt_id, clean_name, entity_type))
                if found:
                    ent_id, strength = found
                    new_links.append(EntityPointsToEvent(
                        entity_id=ent_id, entity_name=entity_name, entity_type=entity_type,
                        next_event_id=evt_id, relationship=rel, strength=strength
                    ))
        else:
            # CHAIN: Link entity to NEXT event
            for i in range(len(occurrences) - 1):
                curr_evt, _ = occurrences[i]
                next_evt, _ = occurrences[i+1]
                
                # Lookup entity from CURRENT event
                found = prod_lookup.get((curr_evt, clean_name, entity_type))
                if found:
                    ent_id, strength = found
                    new_links.append(EntityPointsToEvent(
                        entity_id=ent_id, entity_name=entity_name, entity_type=entity_type,
                        next_event_id=next_evt, relationship=rel, strength=strength
                    ))
                    
    return new_links