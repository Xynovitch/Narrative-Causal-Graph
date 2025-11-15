import uuid
import hashlib
import asyncio
from collections import OrderedDict
from typing import List, Dict, Optional, Any, Set
from .schemas import CEKEvent # Assuming DAGValidator uses CEKEvent

# ----------------------------- Bounded LRU Cache -----------------------------
class BoundedCache:
    """Thread-safe bounded LRU cache with proper async initialization"""
    def __init__(self, max_size: int):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self._lock: Optional[asyncio.Lock] = None
    
    async def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
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

# ----------------------------- Utilities ------------------------------------
def _make_id(prefix: str) -> str:
    """Generate unique ID with given prefix"""
    return f"{prefix}/{uuid.uuid4().hex[:8]}"

def _hash_for_cache(text: str, model: str) -> str:
    """Consistent cache key generation"""
    combined = f"{model}::{text}"
    return hashlib.sha256(combined.encode()).hexdigest()

def _escape_cypher_string(s: str) -> str:
    """Properly escape strings for Cypher queries"""
    if not s:
        return ""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s

def _truncate_safe(text: str, max_length: int = 200) -> str:
    """Safely truncate text without breaking escape sequences"""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    while truncated and truncated.endswith("\\"):
        truncated = truncated[:-1]
    return truncated

def _normalize_weights(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize whyFactor weights to sum to 1.0"""
    if not factors:
        return factors
    
    total = sum(f.get("weight", 0.0) for f in factors)
    if total <= 0:
        even_weight = 1.0 / len(factors)
        for f in factors:
            f["weight"] = round(even_weight, 3)
    else:
        for f in factors:
            f["weight"] = round(f.get("weight", 0.0) / total, 3)
    
    return factors

# ----------------------------- DAG Utilities ---------------------------------
class DAGValidator:
    """Validates DAG properties"""
    
    def __init__(self):
        self.adj_list: Dict[str, Set[str]] = {}
        self.in_degree: Dict[str, int] = {}
        self.event_sequence_map: Dict[str, int] = {}
        self.edge_count = 0
    
    def add_events(self, events: List[CEKEvent]):
        for ev in events:
            self.adj_list[ev.id] = set()
            self.in_degree[ev.id] = 0
            self.event_sequence_map[ev.id] = ev.sequence
    
    def add_edge(self, cause_id: str, effect_id: str) -> bool:
        cause_seq = self.event_sequence_map.get(cause_id)
        effect_seq = self.event_sequence_map.get(effect_id)
        
        if cause_seq is None or effect_seq is None:
            return False
        
        if cause_seq >= effect_seq:
            return False
        
        if effect_id in self.adj_list[cause_id]:
            return False # Edge already exists
        
        self.adj_list[cause_id].add(effect_id)
        self.in_degree[effect_id] = self.in_degree.get(effect_id, 0) + 1
        self.edge_count += 1
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self.adj_list),
            "edges": self.edge_count,
            "max_in_degree": max(self.in_degree.values()) if self.in_degree else 0,
            "max_out_degree": max(len(v) for v in self.adj_list.values()) if self.adj_list else 0
        }