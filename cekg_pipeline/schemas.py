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
    Updated to match both branches with proper field names.
    """
    id: str
    raw_description: str  # Natural language description (from Experimental)
    event_category: str   # Event type category (from Experimental)
    action_type: str      # Canonical action verb
    time_context: Optional[str]      # Temporal reference (from Dynamic)
    location_context: Optional[str]  # Spatial reference (from Dynamic)
    actors: List[str]     # List of actor names
    patients: List[str]   # List of patient names
    chapter: int
    sequence: int = 0
    confidence: float = 1.0
    source_quote: str = ""
    why_factors: List[str] = field(default_factory=list)  # Motivational factors

@dataclass
class EventProducesEntity:
    """Event -[:PRODUCES_X]-> Entity (Event creates/produces entity instance)"""
    event_id: str
    entity_id: str
    entity_name: str
    entity_type: str  # "actor", "patient", "whyfactor", "place"
    relationship: str  # "PRODUCES_ACTOR", "PRODUCES_PATIENT", etc.
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
    """
    Event -[:CAUSES]-> Event
    Updated to use dynamic relation_type instead of fixed enum.
    """
    source_event_id: str
    target_event_id: str
    relation_type: str  # Dynamic string (e.g., "DIRECT_CAUSE", "ENABLES", "PREVENTS")
    mechanism: str      # Explanation of how the causation works
    weight: float       # Strength of causal influence
    confidence: float   # Confidence in this assessment

# ----------------------------- Experimental Features -------------------------

@dataclass
class SemanticLink:
    """
    Event -[:EXPLAINS/CONTRASTS]-> Event
    Non-causal semantic relationships between events.
    NEW from Experimental Features Branch.
    """
    id: str
    source_event_ids: List[str]
    target_event_ids: List[str]
    relation: str  # "explanation", "elaboration", "contrast", "parallelism"
    cue: Optional[List[str]]  # Discourse markers (e.g., ["because", "therefore"])
    confidence: float

@dataclass
class Scene:
    """
    A grouping of events into a narrative scene.
    Scenes cluster events by temporal, spatial, and thematic coherence.
    NEW from Experimental Features Branch.
    """
    id: str
    chapter: int
    included_event_ids: List[str]
    primary_location: Optional[str]
    time_period: Optional[str]
    participants: List[str]
    theme: str      # LLM-generated theme description
    summary: str    # Brief summary of the scene
    confidence: float

# ----------------------------- Generic Graph Schemas -------------------------

@dataclass
class GenericNode:
    """
    A generic node for any graph database.
    Separates graph structure from domain logic.
    """
    uid: str  # The unique ID for this node
    label: str  # The Neo4j label (e.g., "Event", "Agent", "Scene")
    properties: Dict[str, Any]  # All other data as key-value pairs

@dataclass
class GenericRelationship:
    """
    A generic relationship for any graph database.
    Separates graph structure from domain logic.
    """
    start_node_uid: str
    end_node_uid: str
    rel_type: str  # The relationship type (e.g., "CAUSES", "ACTS_IN", "INCLUDES")
    properties: Dict[str, Any]  # All edge properties as key-value pairs