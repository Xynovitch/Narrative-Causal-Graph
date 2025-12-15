"""
Coreference Resolution for Agent Extraction
Resolves pronouns, nicknames, and descriptors to canonical character names.

FIX: Proper resolution that actually deduplicates character variations
"""
import re
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

class CoreferenceResolver:
    """
    Resolves entity mentions to canonical character names
    
    FIX: Now actually performs coreference resolution instead of just filtering
    """
    
    # Pronouns to filter out
    PRONOUNS = {
        'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself',
        'they', 'them', 'their', 'theirs', 'themselves',
        'it', 'its', 'itself',
        'i', 'me', 'my', 'mine', 'myself',
        'we', 'us', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves'
    }
    
    # Generic descriptors to filter
    GENERIC_DESCRIPTORS = {
        'narrator', 'author', 'speaker', 'voice',
        'man', 'woman', 'boy', 'girl', 'child', 'person', 'people',
        'someone', 'something', 'anyone', 'everyone', 'nobody',
        'other', 'another', 'others', 'one'
    }
    
    def __init__(self):
        # Character name registry: {canonical_name: {aliases}}
        self.character_registry: Dict[str, Set[str]] = {}
        # Reverse lookup: {alias_lower: canonical_name}
        self.alias_to_canonical: Dict[str, str] = {}
        # Co-occurrence tracking for smart resolution
        self.cooccurrence_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    
    def is_valid_character_name(self, name: str) -> bool:
        """Check if a name is valid (not pronoun, not generic)"""
        name_lower = name.lower().strip()
        
        # Filter out empty
        if not name_lower or len(name_lower) < 2:
            return False
        
        # Filter out pronouns
        if name_lower in self.PRONOUNS:
            return False
        
        # Filter out generic descriptors
        if name_lower in self.GENERIC_DESCRIPTORS:
            return False
        
        # Filter out possessive pronouns (e.g., "his sister")
        if name_lower.startswith(tuple(self.PRONOUNS)):
            return False
        
        # Filter out articles + generic (e.g., "the man", "a woman")
        if name_lower.startswith(('the ', 'a ', 'an ')):
            base = name_lower.split(' ', 1)[1] if ' ' in name_lower else name_lower
            if base in self.GENERIC_DESCRIPTORS:
                return False
        
        return True
    

    def normalize_character_name(self, name: str) -> str:
        """
        Normalize character names: "Pip (Child)" -> "Pip", "Young Pip" -> "Pip"
        """
        if not name:
            return name
        
        # Remove parentheticals
        name = re.sub(r'\s*\([^)]*\)', '', name)
        
        # Remove age descriptors
        for desc in ['Young', 'Old', 'Little', 'Elder', 'Older', 'Younger']:
            if name.startswith(desc + ' '):
                name = name[len(desc):].strip()
        
        # Remove titles (if not the only name)
        words = name.split()
        if len(words) > 1 and words[0] in ['Mr.', 'Mrs.', 'Miss', 'Ms.', 'Dr.', 'Sir', 'Lady', 'Lord']:
            name = ' '.join(words[1:])
        
        return name.strip()

    def register_character(self, canonical_name: str, aliases: List[str] = None):
        """Register a character with their canonical name and known aliases"""
        if not self.is_valid_character_name(canonical_name):
            return
        
        canonical_name = canonical_name.strip()
        
        if canonical_name not in self.character_registry:
            self.character_registry[canonical_name] = set()
        
        # Add the canonical name as an alias of itself
        self.alias_to_canonical[canonical_name.lower()] = canonical_name
        self.character_registry[canonical_name].add(canonical_name)
        
        # Register aliases
        if aliases:
            for alias in aliases:
                if self.is_valid_character_name(alias):
                    alias = alias.strip()
                    self.character_registry[canonical_name].add(alias)
                    self.alias_to_canonical[alias.lower()] = canonical_name
    
    def resolve(self, mention: str, context: List[str] = None) -> Optional[str]:
        """
        Resolve a mention to its canonical character name.
        
        FIX: Now properly resolves to existing characters instead of auto-registering
        
        Args:
            mention: The character mention to resolve
            context: List of other character mentions in the same event
        
        Returns:
            Canonical character name or None if should be filtered
        """
        # Filter invalid mentions
        if not self.is_valid_character_name(mention):
            return None
        
        mention = mention.strip()
        mention_lower = mention.lower()
        
        # Direct lookup (already registered)
        if mention_lower in self.alias_to_canonical:
            return self.alias_to_canonical[mention_lower]
        
        # Partial match (e.g., "Pip" should match "Philip Pirrip")
        for canonical, aliases in self.character_registry.items():
            canonical_lower = canonical.lower()
            
            # Check if mention is a substring of canonical name
            if mention_lower in canonical_lower:
                # Register as alias and return
                self.register_character(canonical, [mention])
                return canonical
            
            # Check if canonical is a substring of mention (e.g., "Philip Pirrip" exists, mention is "Philip")
            if canonical_lower in mention_lower:
                # Register mention as alias
                self.register_character(canonical, [mention])
                return canonical
            
            # Check against existing aliases
            for alias in aliases:
                alias_lower = alias.lower()
                if mention_lower in alias_lower or alias_lower in mention_lower:
                    self.register_character(canonical, [mention])
                    return canonical
        
        # FIX: Smart fuzzy matching for common variations
        # "pip" vs "Pip", "PIRRIP" vs "Pirrip", etc.
        for canonical in self.character_registry.keys():
            # Exact match ignoring case
            if mention_lower == canonical.lower():
                self.register_character(canonical, [mention])
                return canonical
            
            # First name match (e.g., "Joe" -> "Joe Gargery")
            canonical_parts = canonical.lower().split()
            mention_parts = mention_lower.split()
            
            if mention_parts[0] in canonical_parts:
                # Likely a first name match
                self.register_character(canonical, [mention])
                return canonical
        
        # FIX: If it's a proper name (capitalized) and not found, register as NEW character
        # But DON'T auto-register lowercase or weird names
        if mention[0].isupper() and len(mention.split()) <= 3:
            # This looks like a real character name
            self.register_character(mention)
            return mention
        
        # Otherwise, filter it out (probably a generic descriptor we missed)
        return None
    
    def batch_resolve(self, mentions: List[str], context_window: int = 5) -> List[str]:
        """
        Resolve a batch of mentions from the same narrative context.
        
        Args:
            mentions: List of character mentions
            context_window: How many mentions to consider as context
        
        Returns:
            List of resolved canonical names (filtered)
        """
        resolved = []
        
        for i, mention in enumerate(mentions):
            # Get surrounding context
            start = max(0, i - context_window)
            end = min(len(mentions), i + context_window + 1)
            context = mentions[start:end]
            
            canonical = self.resolve(mention, context)
            if canonical:
                resolved.append(canonical)
        
        # Deduplicate while preserving order
        seen = set()
        result = []
        for name in resolved:
            if name not in seen:
                seen.add(name)
                result.append(name)
        
        return result
    
    def learn_from_cooccurrence(self, event_actors: List[List[str]], min_cooccurrence: int = 3):
        """
        Learn character aliases from co-occurrence patterns.
        If "Philip Pirrip" and "Pip" frequently appear together, link them.
        
        Args:
            event_actors: List of actor lists from multiple events
            min_cooccurrence: Minimum times two names must co-occur
        """
        # Track co-occurrences
        for actors in event_actors:
            # Clean actors first
            valid_actors = [a for a in actors if self.is_valid_character_name(a)]
            
            # Count pairs
            for i, actor1 in enumerate(valid_actors):
                for actor2 in valid_actors[i+1:]:
                    pair = tuple(sorted([actor1.lower(), actor2.lower()]))
                    self.cooccurrence_matrix[pair] += 1
        
        # Find likely aliases (high co-occurrence + name similarity)
        for (name1, name2), count in self.cooccurrence_matrix.items():
            if count >= min_cooccurrence:
                # Check if one is substring of other (e.g., "Pip" in "Philip Pirrip")
                if name1 in name2 or name2 in name1:
                    # Longer name is likely canonical
                    canonical = name2 if len(name2) > len(name1) else name1
                    alias = name1 if len(name1) < len(name2) else name2
                    
                    # Register the alias
                    if canonical in self.alias_to_canonical:
                        canonical = self.alias_to_canonical[canonical]
                    
                    self.register_character(canonical, [alias])
    
    def get_statistics(self) -> Dict[str, any]:
        """Get resolver statistics"""
        return {
            "registered_characters": len(self.character_registry),
            "total_aliases": sum(len(aliases) for aliases in self.character_registry.values()),
            "cooccurrence_pairs": len(self.cooccurrence_matrix)
        }


# Singleton instance
_global_resolver = None

def get_resolver() -> CoreferenceResolver:
    """Get or create the global coreference resolver"""
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = CoreferenceResolver()
    return _global_resolver

def reset_resolver():
    """Reset the global resolver (useful for testing)"""
    global _global_resolver
    _global_resolver = None