from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Set

# ----------------------------- Custom Exceptions -----------------------------
class CEKGError(Exception):
    """Base exception for CEKG processing"""
    pass

class ExtractionError(CEKGError):
    """Event extraction failed"""
    pass

class DAGViolationError(CEKGError):
    """Graph contains cycles"""
    pass

# ----------------------------- Data classes ---------------------------------
@dataclass
class CEKEvent:
    """Event is the central node"""
    id: str
    name: str
    eventType: str
    actionType: str
    source_quote: str
    time: Optional[str]
    location: Optional[str]
    location_id: Optional[str]
    causeWeight: float
    confidence: float
    chapter: int
    sequence: int = 0
    
    def __post_init__(self):
        if not self.name or not self.eventType or not self.actionType:
            raise CEKGError(f"Invalid event: missing required fields")

@dataclass
class EventProducesEntity:
    """Event -[:PRODUCES_X]-> Entity (Event creates/produces entity instance)"""
    event_id: str
    entity_id: str
    entity_name: str
    entity_type: str  # "actor", "patient", "whyfactor", "place"
    relationship: str  # "PRODUCES_ACTOR", "PRODUCES_PATIENT", "PRODUCES_MOTIVATION", "PRODUCES_LOCATION"
    strength: float

@dataclass
class EntityPointsToEvent:
    """Entity -[:ACTS_IN/AFFECTED_IN/MOTIVATES/HOSTS]-> NextEvent"""
    entity_id: str
    entity_name: str
    entity_type: str
    next_event_id: str
    relationship: str  # "ACTS_IN", "AFFECTED_IN", "MOTIVATES", "HOSTS"
    strength: float

@dataclass
class CausalLink:
    """Event -[:CAUSES]-> Event"""
    cause_id: str
    effect_id: str
    relationType: str
    mechanism: str
    sign: str
    weight: float
    confidence: float
    cause_sequence: int
    effect_sequence: int

# --- NEW SCHEMA DATACLASSES ---

@dataclass
class SemanticLink:
    """Event -[:EXPLAINS]-> Event (Non-causal semantic link)"""
    id: str
    source_event_ids: List[str]
    target_event_ids: List[str]
    relation: str # e.g., "explanation", "elaboration"
    cue: Optional[List[str]] # e.g., ["because", "therefore"]
    confidence: float

@dataclass
class Scene:
    """A grouping of events into a narrative scene"""
    id: str
    event_ids: List[str]
    chapter: int
    theme: str # LLM-generated theme
    confidence: float

# --- GENERIC GRAPH SCHEMAS ---

@dataclass
class GenericNode:
    """A generic node for any graph database."""
    uid: str  # The unique ID for this node
    label: str  # The Neo4j label (e.g., "Event", "Agent")
    properties: Dict[str, Any]  # All other data

@dataclass
class GenericRelationship:
    """A generic relationship for any graph database."""
    start_node_uid: str
    end_node_uid: str
    rel_type: str  # The relationship type (e.g., "CAUSES", "ACTS_IN")
    properties: Dict[str, Any]