import uuid
import hashlib
import asyncio
import threading
from collections import OrderedDict
from typing import List, Dict, Optional, Any, Set
from .schemas import CEKEvent

# ----------------------------- Bounded LRU Cache -----------------------------
class BoundedCache:
    """
    Thread-safe bounded LRU cache with proper async initialization.
    
    FIX: Added threading lock to prevent race condition in _get_lock()
    """
    def __init__(self, max_size: int):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self._lock: Optional[asyncio.Lock] = None
        self._init_lock = threading.Lock()  # FIX: Thread-safe initialization
    
    async def _get_lock(self) -> asyncio.Lock:
        """Get or create the asyncio lock in a thread-safe manner"""
        if self._lock is None:
            with self._init_lock:  # FIX: Protect initialization
                if self._lock is None:  # Double-check pattern
                    self._lock = asyncio.Lock()
        return self._lock
    
    async def get(self, key: str) -> Optional[Any]:
        lock = await self._get_lock()
        async with lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    async def set(self, key: str, value: Any) -> None:
        lock = await self._get_lock()
        async with lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    async def size(self) -> int:
        lock = await self._get_lock()
        async with lock:
            return len(self.cache)
    
    async def clear(self) -> None:
        """Clear all cached items"""
        lock = await self._get_lock()
        async with lock:
            self.cache.clear()

# ----------------------------- Utilities ------------------------------------
def _make_id(prefix: str) -> str:
    """Generate unique ID with given prefix"""
    return f"{prefix}/{uuid.uuid4().hex[:8]}"

def _hash_for_cache(text: str, model: str) -> str:
    """Consistent cache key generation"""
    combined = f"{model}::{text}"
    return hashlib.sha256(combined.encode()).hexdigest()

def _escape_cypher_string(s: str) -> str:
    """
    Properly escape strings for Cypher queries.
    
    FIX: Added validation for None input
    """
    if not s or s is None:
        return ""
    
    # Convert to string if not already
    s = str(s)
    
    # Escape special characters
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s

def _truncate_safe(text: str, max_length: int = 200) -> str:
    """
    Safely truncate text without breaking escape sequences.
    
    FIX: Added validation for None and non-string input
    """
    if not text or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    
    # Avoid breaking escape sequences
    while truncated and truncated.endswith("\\"):
        truncated = truncated[:-1]
    
    return truncated

def _normalize_weights(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize whyFactor weights to sum to 1.0.
    
    FIX: Added validation for empty or invalid factors
    """
    if not factors or not isinstance(factors, list):
        return []
    
    # Filter out invalid entries
    valid_factors = [f for f in factors if isinstance(f, dict)]
    
    if not valid_factors:
        return []
    
    total = sum(f.get("weight", 0.0) for f in valid_factors)
    
    if total <= 0:
        # Equal weight distribution
        even_weight = 1.0 / len(valid_factors)
        for f in valid_factors:
            f["weight"] = round(even_weight, 3)
    else:
        # Normalize to sum to 1.0
        for f in valid_factors:
            f["weight"] = round(f.get("weight", 0.0) / total, 3)
    
    return valid_factors

# ----------------------------- DAG Utilities ---------------------------------
class DAGValidator:
    """
    Validates DAG properties and prevents cycles.
    
    FIX: Added better error handling and validation
    """
    
    def __init__(self):
        self.adj_list: Dict[str, Set[str]] = {}
        self.in_degree: Dict[str, int] = {}
        self.event_sequence_map: Dict[str, int] = {}
        self.edge_count = 0
    
    def add_events(self, events: List[CEKEvent]):
        """Add events to the validator"""
        if not events:
            return
        
        for ev in events:
            if not ev or not ev.id:
                continue
            
            self.adj_list[ev.id] = set()
            self.in_degree[ev.id] = 0
            self.event_sequence_map[ev.id] = ev.sequence
    
    def add_edge(self, cause_id: str, effect_id: str) -> bool:
        """
        Add an edge if it doesn't create a cycle.
        
        Returns:
            bool: True if edge was added, False if rejected
        """
        # FIX: Validate inputs
        if not cause_id or not effect_id:
            return False
        
        if cause_id == effect_id:
            return False  # No self-loops
        
        # Check if nodes exist
        cause_seq = self.event_sequence_map.get(cause_id)
        effect_seq = self.event_sequence_map.get(effect_id)
        
        if cause_seq is None or effect_seq is None:
            return False
        
        # Enforce temporal ordering (cause must come before effect)
        if cause_seq >= effect_seq:
            return False
        
        # Check if edge already exists
        if effect_id in self.adj_list[cause_id]:
            return False
        
        # FIX: Check for cycle before adding
        if self._would_create_cycle(cause_id, effect_id):
            return False
        
        # Add edge
        self.adj_list[cause_id].add(effect_id)
        self.in_degree[effect_id] = self.in_degree.get(effect_id, 0) + 1
        self.edge_count += 1
        return True
    
    def _would_create_cycle(self, from_node: str, to_node: str) -> bool:
        """
        Check if adding edge from_node -> to_node would create a cycle.
        Uses DFS to detect if there's already a path from to_node to from_node.
        """
        # If there's a path from to_node to from_node, adding this edge creates a cycle
        visited = set()
        
        def dfs(node: str) -> bool:
            if node == from_node:
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            
            for neighbor in self.adj_list.get(node, set()):
                if dfs(neighbor):
                    return True
            
            return False
        
        return dfs(to_node)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics"""
        max_in = max(self.in_degree.values()) if self.in_degree else 0
        max_out = max(len(v) for v in self.adj_list.values()) if self.adj_list else 0
        
        return {
            "nodes": len(self.adj_list),
            "edges": self.edge_count,
            "max_in_degree": max_in,
            "max_out_degree": max_out,
            "avg_in_degree": round(self.edge_count / len(self.adj_list), 2) if self.adj_list else 0
        }
    
    def validate_dag(self) -> bool:
        """
        Validate that the graph is a DAG using Kahn's algorithm.
        
        Returns:
            bool: True if graph is a valid DAG
        """
        # Copy in-degrees
        in_deg_copy = self.in_degree.copy()
        queue = [node for node, deg in in_deg_copy.items() if deg == 0]
        processed = 0
        
        while queue:
            node = queue.pop(0)
            processed += 1
            
            for neighbor in self.adj_list.get(node, set()):
                in_deg_copy[neighbor] -= 1
                if in_deg_copy[neighbor] == 0:
                    queue.append(neighbor)
        
        # If all nodes were processed, it's a DAG
        return processed == len(self.adj_list)