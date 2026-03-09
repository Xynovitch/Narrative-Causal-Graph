from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

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

# ----------------------------- Core Data Classes -----------------------------

@dataclass
class CEKEvent:
    """
    Event is the central node in the CEKG.
    Now includes theory-based event classification.
    """
    id: str
    raw_description: str
    action_type: str
    time_context: Optional[str]
    location_context: Optional[str]
    actors: List[str]
    patients: List[str]
    chapter: int
    sequence: int = 0
    confidence: float = 1.0
    source_quote: str = ""
    why_factors: List[str] = field(default_factory=list)
    theory: str = "McKee"  # Theory attribution (@McKee or @Truby)

@dataclass
class AgentRole:
    """
    Represents an agent's role in the narrative.
    Maps to AgentTypeDictionary from schema.
    """
    agent_id: str
    agent_name: str
    agent_type: str  # e.g., "PROTAGONIST_HERO", "MORAL_ANTAGONIST", "CRISIS_FORCER"
    theory: str  # "@McKee" or "@Truby"
    events_involved: List[str] = field(default_factory=list)

@dataclass
class EventProducesEntity:
    """Event -[:PRODUCES_X]-> Entity (Event creates/produces entity instance)"""
    event_id: str
    entity_id: str
    entity_name: str
    entity_type: str  # "actor", "patient", "whyfactor", "place", "time"
    relationship: str  # "PRODUCES_ACTOR", "PRODUCES_PATIENT", etc.
    strength: float
    agent_type: Optional[str] = None  # Maps to AgentTypeDictionary if entity_type is actor/patient
    theory: Optional[str] = None  # Theory attribution for agents

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
    """
    Event -[:CAUSES]-> Event
    Now uses theory-based relation types from RelationTypeDictionary.
    """
    source_event_id: str
    target_event_id: str
    relation_type: str  # From RelationTypeDictionary (e.g., "DIRECT_CAUSE", "MORAL_CHALLENGE")
    mechanism: str
    weight: float
    confidence: float
    theory: str = "McKee"  # "@McKee" or "@Truby"
    directionality: str = "uni"  # "uni" or "bi" from schema

@dataclass
class PlaceContext:
    """
    Represents a narrative place with its functional role.
    Maps to PlaceTypeDictionary from schema.
    """
    id: str
    name: str
    place_type: str  # e.g., "MORAL_BATTLEFIELD", "CRISIS_ARENA", "TRANSFORMATION_THRESHOLD"
    narrative_significance: str
    conflict_function: Optional[str] = None
    moral_function: Optional[str] = None
    theory: str = "McKee"
    events: List[str] = field(default_factory=list)

@dataclass
class TimeContext:
    """
    Represents a narrative time period with its structural role.
    Maps to TimeTypeDictionary from schema.
    """
    id: str
    time_type: str  # e.g., "CRISIS_BEAT", "CLIMAX_INTERVAL", "MORALAWAKENING_PHASE"
    narrative_effect: str
    structural_or_moral_function: str
    theory: str = "McKee"
    events: List[str] = field(default_factory=list)

# ----------------------------- Experimental Features -------------------------

@dataclass
class SemanticLink:
    """
    Event -[:EXPLAINS/CONTRASTS]-> Event
    Non-causal semantic relationships between events.
    """
    id: str
    source_event_ids: List[str]
    target_event_ids: List[str]
    relation: str
    cue: Optional[List[str]]
    confidence: float

@dataclass
class Scene:
    """
    A grouping of events into a narrative scene.
    Scenes cluster events by temporal, spatial, and thematic coherence.
    ALL entities and events must belong to at least one scene.
    """
    id: str
    chapter: int
    included_event_ids: List[str]
    primary_location: Optional[str]
    time_period: Optional[str]
    participants: List[str]  # All actors and patients in scene
    theme: str
    summary: str
    confidence: float
    place_type: Optional[str] = None  # Maps to PlaceTypeDictionary
    time_type: Optional[str] = None  # Maps to TimeTypeDictionary
    
    # Extended entity lists (stored but not in core properties)
    all_actors: List[str] = field(default_factory=list)
    all_patients: List[str] = field(default_factory=list)
    all_whyfactors: List[str] = field(default_factory=list)

# ----------------------------- Generic Graph Schemas -------------------------

@dataclass
class GenericNode:
    """
    A generic node for any graph database.
    Separates graph structure from domain logic.
    """
    uid: str
    label: str
    properties: Dict[str, Any]

@dataclass
class GenericRelationship:
    """
    A generic relationship for any graph database.
    Separates graph structure from domain logic.
    """
    start_node_uid: str
    end_node_uid: str
    rel_type: str
    properties: Dict[str, Any]