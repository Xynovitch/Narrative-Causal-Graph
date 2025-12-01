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

# ----------------------------- Data classes ---------------------------------

@dataclass
class CEKEvent:
    """
    Event Node with Attributes.
    Separates raw description from standardized category.
    Context (Time/Location) are attributes, not separate nodes.
    """
    id: str
    
    # Textual Data
    raw_description: str        # Natural language from text
    event_category: str         # Standardized category from Ontology
    
    # Attributes
    action_type: str            # Canonical verb
    time_context: Optional[str]      # e.g., "Morning"
    location_context: Optional[str]  # e.g., "Kitchen"
    
    # Participants
    actors: List[str]
    patients: List[str]
    
    # Metadata
    chapter: int
    sequence: int = 0
    confidence: float = 1.0
    source_quote: str = ""
    
    # WhyFactors are kept as attributes for analysis
    why_factors: List[str] = field(default_factory=list)

@dataclass
class Scene:
    """Narrative container for events."""
    id: str
    chapter: int
    included_event_ids: List[str]
    
    # Scene-level Aggregates
    primary_location: Optional[str]
    time_period: Optional[str]
    participants: List[str]
    
    theme: str
    summary: str
    confidence: float

@dataclass
class CausalLink:
    """Event -[:RELATION_TYPE]-> Event"""
    source_event_id: str
    target_event_id: str
    relation_type: str  # Dynamic String (from Ontology)
    mechanism: str
    weight: float
    confidence: float

@dataclass
class SemanticLink:
    """Event -[:EXPLAINS]-> Event (Non-causal)"""
    id: str
    source_event_ids: List[str]
    target_event_ids: List[str]
    relation: str 
    cue: Optional[List[str]] 
    confidence: float

@dataclass
class EventProducesEntity:
    """
    Link for creation/involvement of an Entity (Actor/WhyFactor) in an Event.
    """
    event_id: str
    entity_id: str
    entity_name: str
    entity_type: str  # "actor", "whyfactor"
    relationship: str 
    strength: float

@dataclass
class EntityPointsToEvent:
    """
    Link for Entity acting in or motivating an Event.
    """
    entity_id: str
    entity_name: str
    entity_type: str
    next_event_id: str
    relationship: str
    strength: float

# --- GENERIC GRAPH SCHEMAS ---

@dataclass
class GenericNode:
    uid: str
    label: str
    properties: Dict[str, Any]

@dataclass
class GenericRelationship:
    start_node_uid: str
    end_node_uid: str
    rel_type: str
    properties: Dict[str, Any]