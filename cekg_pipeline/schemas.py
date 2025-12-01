from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class CEKGError(Exception): pass
class ExtractionError(CEKGError): pass
class DAGViolationError(CEKGError): pass

# NOTE: RelationType Enum is removed to allow dynamic dictionary types.
# You can maintain a list of 'Core' types if validation is needed later.

@dataclass
class CEKEvent:
    id: str
    raw_description: str
    event_category: str
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

@dataclass
class Scene:
    id: str
    chapter: int
    included_event_ids: List[str]
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
    relation_type: str  # Changed from Enum to str for dynamic types
    mechanism: str
    weight: float
    confidence: float

@dataclass
class SemanticLink:
    id: str
    source_event_ids: List[str]
    target_event_ids: List[str]
    relation: str 
    cue: Optional[List[str]] 
    confidence: float

@dataclass
class EventProducesEntity:
    event_id: str
    entity_id: str
    entity_name: str
    entity_type: str 
    relationship: str 
    strength: float

@dataclass
class EntityPointsToEvent:
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