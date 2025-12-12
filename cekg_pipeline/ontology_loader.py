"""
Ontology Loader for CEKG Pipeline
Loads and validates event types, relation types, agent types, etc. from schema files.
"""
import json
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

@dataclass
class EventType:
    """Represents an event type from the schema"""
    name: str
    theory: str
    description: str = ""

@dataclass
class RelationType:
    """Represents a relation type from the schema"""
    name: str
    theory: str
    directionality: str  # "uni" or "bi"
    description: str = ""

@dataclass
class AgentType:
    """Represents an agent type from the schema"""
    name: str
    theory: str
    description: str = ""

class OntologyManager:
    """Manages narrative theory ontologies"""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.event_types: Dict[str, EventType] = {}
        self.relation_types: Dict[str, RelationType] = {}
        self.agent_types: Dict[str, AgentType] = {}
        self.place_types: Dict[str, str] = {}
        self.time_types: Dict[str, str] = {}
        
        if schema_path and os.path.exists(schema_path):
            self._load_from_file(schema_path)
        else:
            self._load_defaults()
    
    def _load_from_file(self, path: str):
        """Load ontologies from JSON schema file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # Load event types
            for evt in schema.get("EventTypeDictionary", []):
                self.event_types[evt["name"]] = EventType(
                    name=evt["name"],
                    theory=evt.get("theory", "@McKee"),
                    description=evt.get("description", "")
                )
            
            # Load relation types
            for rel in schema.get("RelationTypeDictionary", []):
                self.relation_types[rel["name"]] = RelationType(
                    name=rel["name"],
                    theory=rel.get("theory", "@McKee"),
                    directionality=rel.get("directionality", "uni"),
                    description=rel.get("description", "")
                )
            
            # Load agent types
            for agent in schema.get("AgentTypeDictionary", []):
                self.agent_types[agent["name"]] = AgentType(
                    name=agent["name"],
                    theory=agent.get("theory", "@McKee"),
                    description=agent.get("description", "")
                )
            
            # Load place types
            for place in schema.get("PlaceTypeDictionary", []):
                self.place_types[place["name"]] = place.get("theory", "@McKee")
            
            # Load time types
            for time in schema.get("TimeTypeDictionary", []):
                self.time_types[time["name"]] = time.get("theory", "@McKee")
            
            print(f"[ontology] Loaded schema from {path}")
            print(f"[ontology] Event types: {len(self.event_types)}")
            print(f"[ontology] Relation types: {len(self.relation_types)}")
            print(f"[ontology] Agent types: {len(self.agent_types)}")
        except Exception as e:
            print(f"[warning] Failed to load schema from {path}: {e}")
            print("[ontology] Falling back to defaults")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default ontologies when no schema file is provided"""
        # Default McKee event types
        default_events = [
            "PHYSICAL_MOVEMENT", "COMMUNICATION_VERBAL", "INTERNAL_THOUGHT",
            "EMOTIONAL_REACTION", "OBSERVATION", "CONFLICT_PHYSICAL",
            "SOCIAL_INTERACTION", "STATE_CHANGE", "ACQUISITION", "TRAVEL",
            "PHYSICAL_ACTION", "OTHER"
        ]
        for evt in default_events:
            self.event_types[evt] = EventType(evt, "@McKee")
        
        # Default McKee relation types
        default_mckee_relations = [
            "DIRECT_CAUSE", "ENABLES", "PREVENTS", "TRIGGERS", "MOTIVATES",
            "INTERRUPTS", "INHIBITS", "PRECEDES"
        ]
        for rel in default_mckee_relations:
            self.relation_types[rel] = RelationType(rel, "@McKee", "uni")
        
        # Default Truby relation types
        default_truby_relations = [
            "MORAL_CHALLENGE", "WEAKNESS_EXPLOITATION", "DESIRE_FULFILLMENT",
            "REVELATION", "TRANSFORMATION"
        ]
        for rel in default_truby_relations:
            self.relation_types[rel] = RelationType(rel, "@Truby", "uni")
        
        # Default agent types
        default_agents = [
            "PROTAGONIST_HERO", "MORAL_ANTAGONIST", "ALLY_MENTOR",
            "STRUCTURAL_AGENT", "CRISIS_FORCER"
        ]
        for agent in default_agents:
            self.agent_types[agent] = AgentType(agent, "@McKee")
        
        print("[ontology] Loaded default ontologies")
    
    def get_event_type_names(self) -> List[str]:
        """Get all event type names"""
        return list(self.event_types.keys())
    
    def get_relation_type_names(self, theory: Optional[str] = None) -> List[str]:
        """Get relation type names, optionally filtered by theory"""
        if theory:
            theory_normalized = f"@{theory.title()}" if not theory.startswith("@") else theory
            return [name for name, rel in self.relation_types.items() 
                    if rel.theory == theory_normalized]
        return list(self.relation_types.keys())
    
    def get_agent_type_names(self, theory: Optional[str] = None) -> List[str]:
        """Get agent type names, optionally filtered by theory"""
        if theory:
            theory_normalized = f"@{theory.title()}" if not theory.startswith("@") else theory
            return [name for name, agent in self.agent_types.items() 
                    if agent.theory == theory_normalized]
        return list(self.agent_types.keys())
    
    def validate_event_type(self, event_type: str) -> bool:
        """Check if event type is valid"""
        return event_type in self.event_types
    
    def validate_relation_type(self, relation_type: str, theory: Optional[str] = None) -> bool:
        """Check if relation type is valid for the given theory"""
        if relation_type not in self.relation_types:
            return False
        
        if theory:
            theory_normalized = f"@{theory.title()}" if not theory.startswith("@") else theory
            return self.relation_types[relation_type].theory == theory_normalized
        
        return True
    
    def validate_agent_type(self, agent_type: str) -> bool:
        """Check if agent type is valid"""
        return agent_type in self.agent_types
    
    def get_relation_directionality(self, relation_type: str, theory: Optional[str] = None) -> str:
        """Get directionality for a relation type"""
        if relation_type in self.relation_types:
            return self.relation_types[relation_type].directionality
        return "uni"  # Default to unidirectional

# Global singleton instance
_global_ontology_manager: Optional[OntologyManager] = None

def get_ontology_manager(schema_path: Optional[str] = None) -> OntologyManager:
    """Get or create the global ontology manager"""
    global _global_ontology_manager
    if _global_ontology_manager is None:
        _global_ontology_manager = OntologyManager(schema_path)
    return _global_ontology_manager

def reset_ontology_manager():
    """Reset the global ontology manager (useful for testing)"""
    global _global_ontology_manager
    _global_ontology_manager = None