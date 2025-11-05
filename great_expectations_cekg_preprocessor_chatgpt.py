"""
Great Expectations — CEKG preprocessing pipeline (ChatGPT-powered)
EVENT-CENTRIC HIERARCHICAL VERSION - DUAL FLOW

KEY ARCHITECTURAL CHANGES:
- Events PRODUCE entities: Event -[:PRODUCES_ACTOR/PATIENT/MOTIVATION/LOCATION]-> Entity
- Entities POINT TO next events: Entity -[:ACTS_IN/AFFECTED_IN/MOTIVATES/HOSTS]-> NextEvent
- This creates a chain: Event₁ → Entity → Event₂ → Entity → Event₃
- Initial entities exist BEFORE first event (seed nodes)
- Events form causal chains: Event -[:CAUSES]-> Event
- Bidirectional forbidden: only Event→Entity and Entity→Event flows

Usage:
  export OPENAI_API_KEY="sk-..."
  python great_expectations_cekg_dual_flow.py \
    --input "Great Expectations.txt" \
    --out-json ge_preprocessed.json \
    --out-cypher ge_import.cypher
"""
from __future__ import annotations
import os
import re
import json
import time
import uuid
import asyncio
import hashlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Set
import traceback

try:
    import openai
except ImportError:
    openai = None

try:
    import pandas as pd
except ImportError:
    pd = None

# ----------------------------- Configuration ---------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BATCH_SIZE = 5
CAUSAL_BATCH_SIZE = 10
SAMPLE_RATE = 0.5
CACHE_MAX_SIZE = 10000

CONTROLLED_ACTION_ONTOLOGY = {
    "call": "name", "label": "name",
    "see": "perceive", "find": "perceive",
    "think": "imagine", "fancy": "imagine",
    "say": "say", "tell": "say", "announce": "say",
    "ask": "demand", "inquire": "demand",
    "warn": "threaten", "intimidate": "threaten",
    "bring": "give", "offer": "give",
    "go": "move", "leave": "move",
    "eat": "eat", "devour": "eat",
    "vow": "promise", "swear": "promise",
    "strike": "attack", "harm": "attack",
    "tremble": "fear", "cry": "fear",
    "look": "watch", "gaze": "watch",
    "symbolize": "represent", "signify": "represent",
}

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

# ----------------------------- Bounded LRU Cache -----------------------------
class BoundedCache:
    """Thread-safe bounded LRU cache with proper async initialization"""
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
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

event_extraction_cache = BoundedCache()
assessment_cache = BoundedCache()

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

# ----------------------------- ChatGPT Event Extractor ----------------------
def _init_openai_client() -> openai.OpenAI:
    if openai is None:
        raise RuntimeError("openai package not installed. Run `pip install openai`.")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return openai.OpenAI(api_key=OPENAI_API_KEY)

async def extract_events_with_chatgpt_async(
    paragraph: str, 
    chapter_id: int, 
    model: str,
    client: openai.OpenAI
) -> List[Dict[str, Any]]:
    """Use ChatGPT to extract events from a paragraph"""
    cache_key = _hash_for_cache(f"{chapter_id}:{paragraph}", model)
    
    cached = await event_extraction_cache.get(cache_key)
    if cached is not None:
        return cached
    
    # MODIFIED PROMPT: Made whyFactors mandatory and improved location rule
    prompt = f"""You are an expert literary analyst extracting structured events from narrative text.

Extract EVERY distinct event from this paragraph. An event is ANY action, perception, dialogue, emotion, or state change.

Return a JSON array where each event is an object with:

- name: brief description
- eventType: one of ["perception/recognition", "encounter/dialogue", "conflict/threat", "assistance/exchange", "movement/departure", "reflection/emotion", "action", "other"]
- actionType: canonical verb
- actors: array of {{"name": "Pip", "strength": 0.0-1.0}} (use proper names, NEVER pronouns)
- patients: array of {{"name": "Magwitch", "strength": 0.0-1.0}}
- location: The specific place mentioned in the quote (e.g., "churchyard", "the forge"). If not explicitly mentioned, return null.
- time: temporal reference (if any)
- whyFactors: **This is mandatory.** Array of AT LEAST ONE {{"factor": "description", "weight": 0.0-1.0, "category": "Immediate Motivation/Instrumental Cause/Contextual Cause/Symbolic Cause"}}. If no motivation is explicit, infer the most likely "Contextual Cause" (e.g., "fear", "reacting to previous event", "survival instinct", "narrative progression").
- confidence: 0.0-1.0
- quote: the relevant sentence(s) from the text

PARAGRAPH (Chapter {chapter_id}):
{paragraph.strip()}

JSON array:"""

    for attempt in range(3):
        try:
            loop = asyncio.get_event_loop()
            
            def make_request():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=3072,
                )
            
            resp = await loop.run_in_executor(None, make_request)
            text = resp.choices[0].message.content.strip()
            
            if text.startswith("```"):
                lines = text.split('\n')
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()
            
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            if "[" not in text or "]" not in text:
                if "{" in text and "}" in text:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    text = "[" + text[start:end] + "]"
                else:
                    raise ExtractionError(f"No JSON found in response")
            else:
                start = text.find("[")
                end = text.rfind("]") + 1
                text = text[start:end]
            
            # Clean control characters
            cleaned_chars = []
            i = 0
            in_string = False
            escape_next = False
            
            while i < len(text):
                char = text[i]
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    cleaned_chars.append(char)
                    i += 1
                    continue
                
                if char == '\\' and not escape_next:
                    escape_next = True
                    cleaned_chars.append(char)
                    i += 1
                    continue
                
                if escape_next:
                    cleaned_chars.append(char)
                    escape_next = False
                    i += 1
                    continue
                
                if in_string:
                    if char == '\n':
                        cleaned_chars.append('\\n')
                    elif char == '\r':
                        cleaned_chars.append('\\r')
                    elif char == '\t':
                        cleaned_chars.append('\\t')
                    elif ord(char) < 32:
                        pass
                    else:
                        cleaned_chars.append(char)
                else:
                    if ord(char) >= 32 or char in '\n\r\t ':
                        cleaned_chars.append(char)
                
                i += 1
            
            text = ''.join(cleaned_chars)
            
            events = json.loads(text)
            
            if not isinstance(events, list):
                events = [events]
            
            await event_extraction_cache.set(cache_key, events)
            return events
            
        except Exception as e:
            if attempt == 2:
                raise ExtractionError(f"Extraction failed after 3 attempts: {e}")
            await asyncio.sleep(1 + attempt * 2)
    
    raise ExtractionError("Failed to extract events")


async def batch_extract_events(
    paragraphs: List[tuple[str, int]], 
    model: str,
    client: openai.OpenAI
) -> List[List[Dict[str, Any]]]:
    tasks = [
        extract_events_with_chatgpt_async(para, chapter_id, model, client)
        for para, chapter_id in paragraphs
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[batch_extract] Paragraph {i} failed: {result}")
            processed_results.append([])
        else:
            processed_results.append(result)
    
    return processed_results

async def assess_pair_with_chatgpt_async(
    cause_quote: str, 
    effect_quote: str, 
    model: str,
    client: openai.OpenAI
) -> Optional[Dict[str, Any]]:
    cache_key = _hash_for_cache(f"{cause_quote}|||{effect_quote}", model)
    
    cached = await assessment_cache.get(cache_key)
    if cached is not None:
        return cached
    
    prompt = (
        "You are a careful literary causal analyst.\n"
        "Given two event descriptions, return a single JSON object with:\n"
        "- relationType: one of [\"causes\", \"enables\", \"prevents\", \"aggravates\", \"none\"]\n"
        "- mechanism: short explanation (<=50 words)\n"
        "- sign: + or - or 0\n"
        "- weight: float 0.0-1.0\n"
        "- confidence: float 0.0-1.0\n\n"
        f"CAUSE: \"{cause_quote.strip()}\"\n"
        f"EFFECT: \"{effect_quote.strip()}\"\n\n"
        "Return ONLY the JSON object:"
    )

    for attempt in range(3):
        try:
            loop = asyncio.get_event_loop()
            
            def make_request():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
            
            resp = await loop.run_in_executor(None, make_request)
            text = resp.choices[0].message.content.strip()
            
            if text.startswith("```"):
                lines = text.split('\n')
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()
            
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            if "{" not in text or "}" not in text:
                raise ExtractionError(f"No JSON found")
            
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
            
            # Clean control characters (same as above)
            cleaned_chars = []
            i = 0
            in_string = False
            escape_next = False
            
            while i < len(text):
                char = text[i]
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    cleaned_chars.append(char)
                    i += 1
                    continue
                
                if char == '\\' and not escape_next:
                    escape_next = True
                    cleaned_chars.append(char)
                    i += 1
                    continue
                
                if escape_next:
                    cleaned_chars.append(char)
                    escape_next = False
                    i += 1
                    continue
                
                if in_string:
                    if char == '\n':
                        cleaned_chars.append('\\n')
                    elif char == '\r':
                        cleaned_chars.append('\\r')
                    elif char == '\t':
                        cleaned_chars.append('\\t')
                    elif ord(char) < 32:
                        pass
                    else:
                        cleaned_chars.append(char)
                else:
                    if ord(char) >= 32 or char in '\n\r\t ':
                        cleaned_chars.append(char)
                
                i += 1
            
            text = ''.join(cleaned_chars)
            
            obj = json.loads(text)
            await assessment_cache.set(cache_key, obj)
            return obj
            
        except Exception as e:
            if attempt == 2:
                return None
            await asyncio.sleep(1 + attempt * 2)
    
    return None

async def batch_assess_pairs(
    pairs: List[tuple[str, str, str, str]], 
    model: str,
    client: openai.OpenAI
) -> List[Optional[Dict[str, Any]]]:
    tasks = [
        assess_pair_with_chatgpt_async(cause_quote, effect_quote, model, client)
        for cause_quote, effect_quote, _, _ in pairs
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)

# ----------------------------- CEKG Preprocessor ----------------------------
class CEKGPreprocessor:
    def __init__(self, openai_model: Optional[str] = None):
        self.events: List[CEKEvent] = []
        self.causal_links: List[CausalLink] = []
        self.event_produces: List[EventProducesEntity] = []
        self.entity_points_to: List[EntityPointsToEvent] = []
        self.dag_validator = DAGValidator()
        
        # Track entity occurrences for linking
        self.entity_occurrences: Dict[str, List[tuple[str, int]]] = defaultdict(list)  # entity_name -> [(event_id, sequence)]
        
        if not OPENAI_API_KEY or openai is None:
            raise RuntimeError("ChatGPT API is required. Set OPENAI_API_KEY environment variable.")
        
        self.openai_model = openai_model or OPENAI_MODEL
        self.client = _init_openai_client()

    def load_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        
    def split_chapters(self, text: str) -> List[tuple[int, str]]:
        patterns = [
            r"(?m)^Chapter\s+[IVXLCDM]+\.\s*$",
            r"(?m)^CHAPTER\s+[IVXLCDM]+\.?\s*$",
            r"(?m)^Chapter\s+\d+\.?\s*$",
            r"(?m)^CHAPTER\s+\d+\.?\s*$",
            r"(?im)^chapter\s+[IVXLCDM\d]+\.?\s*$",
        ]
        
        parts = None
        
        for i, pattern in enumerate(patterns):
            parts = re.split(pattern, text)
            if len(parts) > 1:
                print(f"[chapter split] Matched pattern {i+1}: found {len(parts)-1} chapters")
                break
        
        if parts is None or len(parts) <= 1:
            print("[chapter split] No chapter markers found, using paragraph splitting")
            paras = [p.strip() for p in text.split('\n\n') if p.strip()]
            return list(enumerate(paras, start=1))
        
        chapters = []
        for idx, part in enumerate(parts):
            cleaned = part.strip()
            if cleaned:
                chapters.append((idx + 1, cleaned))
        
        print(f"[chapter split] Successfully split into {len(chapters)} chapters")
        return chapters

    def split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
        
        chunks = []
        for para in paragraphs:
            if len(para) > 800:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = []
                current_length = 0
                
                for sent in sentences:
                    if current_length + len(sent) > 600 and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_length = len(sent)
                    else:
                        current_chunk.append(sent)
                        current_length += len(sent)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
            else:
                chunks.append(para)
        
        return chunks if chunks else paragraphs

    async def process_chapter_events(
        self, 
        chapter_id: int, 
        chapter_text: str, 
        batch_size: int = BATCH_SIZE
    ) -> tuple[List[CEKEvent], List[EventProducesEntity]]:
        """Process a chapter and extract events with DUAL FLOW"""
        paragraphs = self.split_into_paragraphs(chapter_text)
        print(f"[chapter {chapter_id}] Processing {len(paragraphs)} paragraphs...")
        
        all_events = []
        all_produces = []
        
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i:i+batch_size]
            batch_with_chapter = [(para, chapter_id) for para in batch]
            
            try:
                events_batch = await batch_extract_events(
                    batch_with_chapter, 
                    self.openai_model,
                    self.client
                )
                
                batch_event_count = 0
                for para_events in events_batch:
                    if isinstance(para_events, Exception):
                        continue
                        
                    for event_data in para_events:
                        try:
                            why_factors = _normalize_weights(event_data.get("whyFactors", []))
                            # Ensure whyFactors exists due to new prompt
                            if not why_factors:
                                print(f"[warning] LLM failed to provide mandatory whyFactor for event: {event_data.get('name')}")
                                # Add a default one if LLM fails
                                why_factors = [{"factor": "narrative progression", "weight": 1.0, "category": "Contextual Cause"}]

                            cause_weight = sum(wf.get("weight", 0) for wf in why_factors)
                            
                            action_type = event_data.get("actionType", "")
                            action_type = CONTROLLED_ACTION_ONTOLOGY.get(action_type.lower(), action_type)
                            
                            # --- FIX FOR LOCATION ---
                            raw_location = event_data.get("location")
                            # Normalize the location name
                            location = raw_location.strip() if raw_location else None
                            # --- END FIX ---
                            
                            location_id = f"place/{location.lower().replace(' ', '_')}" if location else None
                            
                            event = CEKEvent(
                                id=_make_id("event"),
                                name=event_data.get("name", ""),
                                eventType=event_data.get("eventType", "other"),
                                actionType=action_type,
                                source_quote=event_data.get("quote", ""),
                                time=event_data.get("time"),
                                location=location, # <-- Use normalized location
                                location_id=location_id,
                                causeWeight=cause_weight,
                                confidence=float(event_data.get("confidence", 0.7)),
                                chapter=chapter_id,
                                sequence=len(self.events) + len(all_events)
                            )
                            all_events.append(event)
                            
                            # Event PRODUCES actors
                            actors = event_data.get("actors", [])
                            if not actors and event_data.get("actor"):
                                actors = [{"name": event_data["actor"], "strength": 1.0}]
                            
                            for actor in actors:
                                # --- FIX FOR ACTOR ---
                                raw_actor_name = actor.get("name")
                                if raw_actor_name:
                                    # Normalize the name
                                    actor_name = raw_actor_name.strip()
                                    actor_key = actor_name.lower() # Key for dictionary
                                    
                                    actor_id = _make_id(f"agent_{actor_key.replace(' ', '_')}")
                                    all_produces.append(EventProducesEntity(
                                        event_id=event.id,
                                        entity_id=actor_id,
                                        entity_name=actor_name, # <-- Store normalized name
                                        entity_type="actor",
                                        relationship="PRODUCES_ACTOR",
                                        strength=float(actor.get("strength", 1.0))
                                    ))
                                    # Track occurrence for later linking
                                    self.entity_occurrences[f"actor:{actor_key}"].append((event.id, event.sequence))
                                # --- END FIX ---
                            
                            # Event PRODUCES patients
                            patients = event_data.get("patients", [])
                            if not patients and event_data.get("patient"):
                                patients = [{"name": event_data["patient"], "strength": 1.0}]
                            
                            for patient in patients:
                                # --- FIX FOR PATIENT ---
                                raw_patient_name = patient.get("name")
                                if raw_patient_name:
                                    # Normalize the name
                                    patient_name = raw_patient_name.strip()
                                    patient_key = patient_name.lower() # Key for dictionary

                                    patient_id = _make_id(f"agent_{patient_key.replace(' ', '_')}")
                                    all_produces.append(EventProducesEntity(
                                        event_id=event.id,
                                        entity_id=patient_id,
                                        entity_name=patient_name, # <-- Store normalized name
                                        entity_type="patient",
                                        relationship="PRODUCES_PATIENT",
                                        strength=float(patient.get("strength", 1.0))
                                    ))
                                    self.entity_occurrences[f"patient:{patient_key}"].append((event.id, event.sequence))
                                # --- END FIX ---
                            
                            # Event PRODUCES whyfactors
                            for wf in why_factors:
                                # --- FIX FOR WHYFACTOR ---
                                raw_factor = wf.get('factor')
                                if raw_factor:
                                    # Normalize the factor
                                    factor_name = raw_factor.strip()
                                    factor_key = factor_name.lower() # Key for dictionary

                                    wf_id = _make_id(f"why_{factor_key[:30].replace(' ', '_')}")
                                    all_produces.append(EventProducesEntity(
                                        event_id=event.id,
                                        entity_id=wf_id,
                                        entity_name=factor_name, # <-- Store normalized name
                                        entity_type="whyfactor",
                                        relationship="PRODUCES_MOTIVATION",
                                        strength=wf['weight']
                                    ))
                                    self.entity_occurrences[f"whyfactor:{factor_key}"].append((event.id, event.sequence))
                                # --- END FIX ---
                            
                            # Event PRODUCES location
                            if location: # <-- Already normalized from above
                                # --- FIX FOR LOCATION ---
                                location_key = location.lower() # Key for dictionary
                                place_id = _make_id(f"place_{location_key.replace(' ', '_')}")
                                all_produces.append(EventProducesEntity(
                                    event_id=event.id,
                                    entity_id=place_id,
                                    entity_name=location, # <-- Store normalized name
                                    entity_type="place",
                                    relationship="PRODUCES_LOCATION",
                                    strength=0.8
                                ))
                                self.entity_occurrences[f"place:{location_key}"].append((event.id, event.sequence))
                            # --- END FIX ---
                            
                            batch_event_count += 1
                            
                        except Exception as e:
                            print(f"[warning] Failed to process event: {e}")
                            traceback.print_exc()
                            continue
                
                print(f"[chapter {chapter_id}] Batch {i//batch_size + 1}: extracted {batch_event_count} events")
                
            except Exception as e:
                print(f"[error] Batch {i//batch_size + 1} failed: {e}")
                traceback.print_exc()
        
        return all_events, all_produces

    # =========================================================================
    # MODIFIED FUNCTION
    # This now propagates Actors, Locations, AND WhyFactors
    # =========================================================================
    def propagate_context(self):
        """
        Pass 2: Iterate through all events and propagate location, actor,
        and whyfactor context to fill in the blanks.
        """
        print("[context] Propagating stateful context (locations, actors, whyfactors)...")
        current_location_name = None
        # {actor_key: (actor_name, actor_type, relationship)}
        current_actors = {}
        # [(factor_name, factor_type, relationship, strength), ...]
        current_whyfactors = []

        newly_produced = [] # To store new links we create
        
        # Build a lookup for explicitly produced entities
        prods_by_event = defaultdict(list)
        for prod in self.event_produces:
            prods_by_event[prod.event_id].append(prod)

        # IMPORTANT: Assumes self.events is already sorted by sequence
        for event in self.events:
            
            explicit_actors = {}
            explicit_location_name = None
            explicit_whyfactors = []
            
            # Check for entities explicitly extracted for this event
            for prod in prods_by_event[event.id]:
                if prod.entity_type == 'actor' or prod.entity_type == 'patient':
                    key = prod.entity_name.lower()
                    explicit_actors[key] = (prod.entity_name, prod.entity_type, prod.relationship)
                elif prod.entity_type == 'place':
                    explicit_location_name = prod.entity_name
                elif prod.entity_type == 'whyfactor':
                    explicit_whyfactors.append((prod.entity_name, 'whyfactor', 'PRODUCES_MOTIVATION', prod.strength))

            
            # 1. Propagate Location
            if explicit_location_name:
                # This event sets a new location
                current_location_name = explicit_location_name
            elif current_location_name:
                # This event is missing a location. Assign the current one.
                event.location = current_location_name # Update event object
                loc_key = current_location_name.lower()
                new_loc_id = _make_id(f"place_{loc_key.replace(' ', '_')}")
                event.location_id = new_loc_id # Update event object
                
                new_prod = EventProducesEntity(
                    event_id=event.id,
                    entity_id=new_loc_id,
                    entity_name=current_location_name,
                    entity_type="place",
                    relationship="PRODUCES_LOCATION",
                    strength=0.5 # Lower strength for inferred context
                )
                newly_produced.append(new_prod)
                self.entity_occurrences[f"place:{loc_key}"].append((event.id, event.sequence))

            # 2. Propagate Actors
            if explicit_actors:
                # This event defines a new set of actors
                current_actors = explicit_actors
            elif current_actors:
                # This event is missing actors. Assign the current ones.
                for actor_key, (actor_name, actor_type, actor_rel) in current_actors.items():
                    new_actor_id = _make_id(f"agent_{actor_key.replace(' ', '_')}")
                    new_prod = EventProducesEntity(
                        event_id=event.id,
                        entity_id=new_actor_id,
                        entity_name=actor_name,
                        entity_type=actor_type,
                        relationship=actor_rel,
                        strength=0.5 # Lower strength for inferred context
                    )
                    newly_produced.append(new_prod)
                    self.entity_occurrences[f"{actor_type}:{actor_key}"].append((event.id, event.sequence))

            # 3. Propagate WhyFactors
            if explicit_whyfactors:
                # This event defines a new set of whyfactors
                current_whyfactors = explicit_whyfactors
            elif current_whyfactors:
                # This event is missing whyfactors. Assign the current ones.
                for factor_name, factor_type, factor_rel, factor_strength in current_whyfactors:
                    factor_key = factor_name.lower()
                    new_factor_id = _make_id(f"why_{factor_key[:30].replace(' ', '_')}")
                    new_prod = EventProducesEntity(
                        event_id=event.id,
                        entity_id=new_factor_id,
                        entity_name=factor_name,
                        entity_type=factor_type,
                        relationship=factor_rel,
                        strength=factor_strength # Propagate original strength
                    )
                    newly_produced.append(new_prod)
                    self.entity_occurrences[f"whyfactor:{factor_key}"].append((event.id, event.sequence))
        
        # Add all new links to the main list
        self.event_produces.extend(newly_produced)
        print(f"[context] Propagated context, created {len(newly_produced)} new entity links.")
        
        # Re-sort all occurrence lists since we added new ones in parallel
        for key in self.entity_occurrences:
            self.entity_occurrences[key].sort(key=lambda x: x[1])
            
    # =========================================================================
    # END OF MODIFIED FUNCTION
    # =========================================================================

    def create_entity_to_event_links(self):
        """Create Entity -[:X]-> NextEvent links based on entity occurrences"""
        print("[linking] Creating entity→event links...")
        
        new_links = []
        
        for entity_key, occurrences in self.entity_occurrences.items():
            # Sort by sequence (redundant if done in propagate_context, but safe)
            occurrences.sort(key=lambda x: x[1])
            
            # Parse entity type and name
            try:
                entity_type, entity_name = entity_key.split(":", 1)
                target_name_lower = entity_name.strip().lower()
            except ValueError:
                print(f"[linking] Skipping malformed entity key: {entity_key}")
                continue
            
            # Create links from each occurrence to the next
            for i in range(len(occurrences) - 1):
                current_event_id, current_seq = occurrences[i]
                next_event_id, next_seq = occurrences[i + 1]
                
                entity_id = None
                strength = 0.5  # Default strength
                
                # Find the entity_id and strength produced by the current event
                for prod in self.event_produces:
                    if not prod.entity_name:
                        continue
                    
                    prod_name = prod.entity_name.strip().lower()
                    
                    # Match on event, type, and name
                    if prod.event_id == current_event_id and \
                       prod.entity_type == entity_type and \
                       prod_name == target_name_lower:
                        
                        entity_id = prod.entity_id
                        strength = prod.strength  # <-- CAPTURE STRENGTH HERE
                        break # Found the match
                
                if entity_id:
                    # Determine relationship type based on the entity type
                    if entity_type == "actor":
                        rel_type = "ACTS_IN"
                    elif entity_type == "patient":
                        rel_type = "AFFECTED_IN"
                    elif entity_type == "whyfactor":
                        rel_type = "MOTIVATES"
                    elif entity_type == "place":
                        rel_type = "HOSTS"
                    else:
                        continue
                    
                    # Add the new Entity->Event link
                    new_links.append(EntityPointsToEvent(
                        entity_id=entity_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        next_event_id=next_event_id,
                        relationship=rel_type,
                        strength=strength
                    ))
        
        self.entity_points_to = new_links
        print(f"[linking] Created {len(self.entity_points_to)} entity→event links")

    # =========================================================================
    # MODIFIED FUNCTION
    # This now runs multiple passes for long-range causality
    # =========================================================================
    # =========================================================================
    # MODIFIED FUNCTION
    # This now runs multiple passes for long-range causality
    # =========================================================================
    async def _batch_causal_linking(
            self, 
            window: int = 4, 
            sample_rate: float = SAMPLE_RATE, 
            batch_size: int = CAUSAL_BATCH_SIZE
        ):
            """Link events causally"""
            import random
            
            self.dag_validator.add_events(self.events)
            print(f"[dag] Initialized DAG validator with {len(self.events)} events")
            
            # Use a set of (cause_id, effect_id) tuples to prevent duplicates
            pairs_to_assess_set: Set[tuple[str, str]] = set()

            # --- PASS 1: LOCAL WINDOW ---
            # Checks for immediate, local causality
            print("[causal linking] Pass 1/3: Assessing local window...")
            for i, ev in enumerate(self.events):
                start = max(0, i - window)
                for j in range(start, i):
                    cand = self.events[j]
                    if cand.sequence >= ev.sequence:
                        continue
                    pairs_to_assess_set.add((cand.id, ev.id))

            # --- PASS 2: SHARED ACTOR/PATIENT ---
            # Links far-off events that share the same character
            print("[causal linking] Pass 2/3: Assessing shared actors...")
            for entity_key, occurrences in self.entity_occurrences.items():
                if not (entity_key.startswith("actor:") or entity_key.startswith("patient:")):
                    continue
                
                # Link each event to the *next* event with the same actor
                for i in range(len(occurrences) - 1):
                    cause_event_id, _ = occurrences[i]
                    effect_event_id, _ = occurrences[i+1]
                    
                    # =================== FIX IS HERE ===================
                    cause_seq = self.dag_validator.event_sequence_map.get(cause_event_id)
                    effect_seq = self.dag_validator.event_sequence_map.get(effect_event_id)
                    # ================= END OF FIX ==================
                    
                    if cause_seq is not None and effect_seq is not None and cause_seq < effect_seq:
                        pairs_to_assess_set.add((cause_event_id, effect_event_id))

            # --- PASS 3: SHARED LOCATION ---
            # Links far-off events that share the same location
            print("[causal linking] Pass 3/3: Assessing shared locations...")
            for entity_key, occurrences in self.entity_occurrences.items():
                if not entity_key.startswith("place:"):
                    continue
                
                # Link each event to the *next* event in the same location
                for i in range(len(occurrences) - 1):
                    cause_event_id, _ = occurrences[i]
                    effect_event_id, _ = occurrences[i+1]

                    # =================== FIX IS HERE ===================
                    cause_seq = self.dag_validator.event_sequence_map.get(cause_event_id)
                    effect_seq = self.dag_validator.event_sequence_map.get(effect_event_id)
                    # ================= END OF FIX ==================

                    if cause_seq is not None and effect_seq is not None and cause_seq < effect_seq:
                        pairs_to_assess_set.add((cause_event_id, effect_event_id))

            # --- Convert set of IDs back into list of quotes for the LLM ---
            print(f"[causal linking] Found {len(pairs_to_assess_set)} unique candidate pairs.")
            
            # Create a quick lookup map for events
            event_map_by_id = {ev.id: ev for ev in self.events}
            
            pairs_to_assess = []
            for cause_id, effect_id in pairs_to_assess_set:
                cause_ev = event_map_by_id.get(cause_id)
                effect_ev = event_map_by_id.get(effect_id)
                
                if cause_ev and effect_ev:
                    pairs_to_assess.append((
                        cause_ev.source_quote,
                        effect_ev.source_quote,
                        cause_id,
                        effect_id
                    ))
            
            print(f"[causal linking] Assessing {len(pairs_to_assess)} final pairs via LLM...")
            
            accepted_edges = 0
            rejected_edges = 0
            
            for batch_start in range(0, len(pairs_to_assess), batch_size):
                batch = pairs_to_assess[batch_start:batch_start + batch_size]
                assessments = await batch_assess_pairs(batch, self.openai_model, self.client)
                
                for (cause_quote, effect_quote, cause_id, effect_id), assessment in zip(batch, assessments):
                    if isinstance(assessment, Exception) or assessment is None:
                        continue
                    
                    if assessment.get("relationType") != "none":
                        if not self.dag_validator.add_edge(cause_id, effect_id):
                            rejected_edges += 1
                            continue
                        
                        # =================== FIX IS HERE ===================
                        cause_seq = self.dag_validator.event_sequence_map.get(cause_id)
                        effect_seq = self.dag_validator.event_sequence_map.get(effect_id)
                        # ================= END OF FIX ==================
                        
                        if cause_seq is None or effect_seq is None:
                            rejected_edges += 1
                            continue
                        
                        link = CausalLink(
                            cause_id=cause_id,
                            effect_id=effect_id,
                            relationType=assessment.get("relationType", "causes"),
                            mechanism=assessment.get("mechanism", ""),
                            sign=assessment.get("sign", "0"),
                            weight=float(assessment.get("weight", 0.0)),
                            confidence=float(assessment.get("confidence", 0.0)),
                            cause_sequence=cause_seq,
                            effect_sequence=effect_seq
                        )
                        self.causal_links.append(link)
                        accepted_edges += 1
                
                print(f"[causal] Progress: {min(batch_start + batch_size, len(pairs_to_assess))}/{len(pairs_to_assess)} "
                    f"(accepted: {accepted_edges}, rejected: {rejected_edges})")
            
            print(f"[dag] Final edge statistics: {accepted_edges} accepted, {rejected_edges} rejected")
            
            stats = self.dag_validator.get_stats()
            print(f"[dag] ✓ DAG validated: {stats['nodes']} nodes, {stats['edges']} edges")
            print(f"[dag] Max in-degree: {stats['max_in_degree']}, Max out-degree: {stats['max_out_degree']}")
    # =========================================================================
    # END OF MODIFIED FUNCTION
    # =========================================================================

    def build_jsonld(self) -> Dict[str, Any]:
        """Build JSON-LD representation with DUAL FLOW structure"""
        g = []
        
        # Events
        for ev in self.events:
            event_dict = asdict(ev)
            event_dict["@id"] = event_dict.pop("id")
            event_dict["type"] = "Event"
            g.append(event_dict)
        
        # Event → Entity (production)
        for prod in self.event_produces:
            g.append({
                "@id": f"{prod.event_id}__{prod.relationship}__{prod.entity_id}",
                "type": "EventProducesEntity",
                "from": prod.event_id,
                "to": prod.entity_id,
                "entity_name": prod.entity_name,
                "entity_type": prod.entity_type,
                "relationship": prod.relationship,
                "strength": prod.strength
            })
        
        # Entity → Event (pointing to next)
        for ept in self.entity_points_to:
            g.append({
                "@id": f"{ept.entity_id}__{ept.relationship}__{ept.next_event_id}",
                "type": "EntityPointsToEvent",
                "from": ept.entity_id,
                "to": ept.next_event_id,
                "entity_name": ept.entity_name,
                "entity_type": ept.entity_type,
                "relationship": ept.relationship,
                "strength": ept.strength
            })
        
        # Event → Event (causal)
        for link in self.causal_links:
            g.append({
                "@id": f"{link.cause_id}__CAUSES__{link.effect_id}",
                "type": "CausalEdge",
                "from": link.cause_id,
                "to": link.effect_id,
                "relationType": link.relationType,
                "mechanism": link.mechanism,
                "sign": link.sign,
                "weight": link.weight,
                "confidence": link.confidence,
                "cause_sequence": link.cause_sequence,
                "effect_sequence": link.effect_sequence
            })
        
        return {"@graph": g}

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.build_jsonld(), f, ensure_ascii=False, indent=2)

    def export_neo4j_cypher(self, path: str):
        """Export to Neo4j Cypher with DUAL FLOW structure"""
        # Collect unique entities
        entities_by_type = defaultdict(dict)
        for prod in self.event_produces:
            entities_by_type[prod.entity_type][prod.entity_id] = prod.entity_name
        
        lines = []
        lines.append("// ============================================================")
        lines.append("// EVENT-CENTRIC CEKG - DUAL FLOW")
        lines.append("// Event₁ -[:PRODUCES_X]-> Entity -[:X_IN]-> Event₂")
        lines.append("// Events produce entities, entities point to next events")
        lines.append("// ============================================================\n")
        
        # 1. Create Event nodes
        lines.append("// 1. CREATE EVENT NODES")
        for ev in self.events:
            escaped_quote = _truncate_safe(_escape_cypher_string(ev.source_quote), 300)
            props = {
                "id": ev.id,
                "name": _escape_cypher_string(ev.name),
                "eventType": ev.eventType,
                "actionType": ev.actionType,
                "source_quote": escaped_quote,
                "causeWeight": ev.causeWeight or 0.0,
                "confidence": ev.confidence,
                "sequence": ev.sequence,
                "chapter": ev.chapter,
                "time": _escape_cypher_string(ev.time) if ev.time else "",
                "location": _escape_cypher_string(ev.location) if ev.location else ""
            }
            props_str = ", ".join([
                f"{k}: \"{v}\"" if isinstance(v, str) and v else f"{k}: {v}" if not isinstance(v, str) else f"{k}: \"\""
                for k, v in props.items()
            ])
            lines.append(f"CREATE (e{ev.sequence}:Event {{{props_str}}});")
        
        lines.append("")
        
        # 2. Create Agent nodes (actors + patients)
        lines.append("// 2. CREATE AGENT NODES (unique instances)")
        all_agents = {}
        all_agents.update(entities_by_type.get("actor", {}))
        all_agents.update(entities_by_type.get("patient", {}))
        for agent_id, agent_name in all_agents.items():
            safe_name = _escape_cypher_string(agent_name)
            lines.append(f"MERGE (a:Agent {{id: \"{agent_id}\", name: \"{safe_name}\"}});")
        
        lines.append("")
        
        # 3. Create WhyFactor nodes
        lines.append("// 3. CREATE WHYFACTOR NODES (unique instances)")
        for wf_id, wf_name in entities_by_type.get("whyfactor", {}).items():
            safe_name = _escape_cypher_string(wf_name)
            lines.append(f"MERGE (w:WhyFactor {{id: \"{wf_id}\", factor: \"{safe_name}\"}});")
        
        lines.append("")
        
        # 4. Create Place nodes
        lines.append("// 4. CREATE PLACE NODES (unique instances)")
        for place_id, place_name in entities_by_type.get("place", {}).items():
            safe_name = _escape_cypher_string(place_name)
            lines.append(f"MERGE (p:Place {{id: \"{place_id}\", name: \"{safe_name}\"}});")
        
        lines.append("")
        lines.append("// ============================================================")
        lines.append("// DUAL FLOW: Event → Entity (production)")
        lines.append("// ============================================================\n")
        
        # 5. Event -[:PRODUCES_ACTOR]-> Agent
        lines.append("// 5. EVENT PRODUCES ACTOR")
        for prod in self.event_produces:
            if prod.entity_type == "actor":
                lines.append(f"MATCH (e:Event {{id: \"{prod.event_id}\"}}), (a:Agent {{id: \"{prod.entity_id}\"}}) "
                            f"MERGE (e)-[:PRODUCES_ACTOR {{strength: {prod.strength}}}]->(a);")
        
        lines.append("")
        
        # 6. Event -[:PRODUCES_PATIENT]-> Agent
        lines.append("// 6. EVENT PRODUCES PATIENT")
        for prod in self.event_produces:
            if prod.entity_type == "patient":
                lines.append(f"MATCH (e:Event {{id: \"{prod.event_id}\"}}), (a:Agent {{id: \"{prod.entity_id}\"}}) "
                            f"MERGE (e)-[:PRODUCES_PATIENT {{strength: {prod.strength}}}]->(a);")
        
        lines.append("")
        
        # 7. Event -[:PRODUCES_MOTIVATION]-> WhyFactor
        lines.append("// 7. EVENT PRODUCES MOTIVATION")
        for prod in self.event_produces:
            if prod.entity_type == "whyfactor":
                lines.append(f"MATCH (e:Event {{id: \"{prod.event_id}\"}}), (w:WhyFactor {{id: \"{prod.entity_id}\"}}) "
                            f"MERGE (e)-[:PRODUCES_MOTIVATION {{weight: {prod.strength}}}]->(w);")
        
        lines.append("")
        
        # 8. Event -[:PRODUCES_LOCATION]-> Place
        lines.append("// 8. EVENT PRODUCES LOCATION")
        for prod in self.event_produces:
            if prod.entity_type == "place":
                lines.append(f"MATCH (e:Event {{id: \"{prod.event_id}\"}}), (p:Place {{id: \"{prod.entity_id}\"}}) "
                            f"MERGE (e)-[:PRODUCES_LOCATION {{specificity: {prod.strength}}}]->(p);")
        
        lines.append("")
        lines.append("// ============================================================")
        lines.append("// DUAL FLOW: Entity → Event (pointing to next)")
        lines.append("// ============================================================\n")
        
        # 9. Agent -[:ACTS_IN]-> Event
        lines.append("// 9. AGENT ACTS IN NEXT EVENT")
        for ept in self.entity_points_to:
            if ept.relationship == "ACTS_IN":
                lines.append(f"MATCH (a:Agent {{id: \"{ept.entity_id}\"}}), (e:Event {{id: \"{ept.next_event_id}\"}}) "
                            f"MERGE (a)-[:ACTS_IN {{strength: {ept.strength}}}]->(e);")
        
        lines.append("")
        
        # 10. Agent -[:AFFECTED_IN]-> Event
        lines.append("// 10. AGENT AFFECTED IN NEXT EVENT")
        for ept in self.entity_points_to:
            if ept.relationship == "AFFECTED_IN":
                lines.append(f"MATCH (a:Agent {{id: \"{ept.entity_id}\"}}), (e:Event {{id: \"{ept.next_event_id}\"}}) "
                            f"MERGE (a)-[:AFFECTED_IN {{strength: {ept.strength}}}]->(e);")
        
        lines.append("")
        
        # 11. WhyFactor -[:MOTIVATES]-> Event
        lines.append("// 11. WHYFACTOR MOTIVATES NEXT EVENT")
        for ept in self.entity_points_to:
            if ept.relationship == "MOTIVATES":
                lines.append(f"MATCH (w:WhyFactor {{id: \"{ept.entity_id}\"}}), (e:Event {{id: \"{ept.next_event_id}\"}}) "
                            f"MERGE (w)-[:MOTIVATES {{weight: {ept.strength}}}]->(e);")
        
        lines.append("")
        
        # 12. Place -[:HOSTS]-> Event
        lines.append("// 12. PLACE HOSTS NEXT EVENT")
        for ept in self.entity_points_to:
            if ept.relationship == "HOSTS":
                lines.append(f"MATCH (p:Place {{id: \"{ept.entity_id}\"}}), (e:Event {{id: \"{ept.next_event_id}\"}}) "
                            f"MERGE (p)-[:HOSTS {{specificity: {ept.strength}}}]->(e);")
        
        lines.append("")
        
        # 13. Event -[:FOLLOWS]-> Event
        lines.append("// 13. EVENT FOLLOWS EVENT (temporal sequence)")
        for i in range(len(self.events) - 1):
            ev1 = self.events[i]
            ev2 = self.events[i + 1]
            if ev1.chapter == ev2.chapter:
                lines.append(f"MATCH (e1:Event {{id: \"{ev1.id}\"}}), (e2:Event {{id: \"{ev2.id}\"}}) "
                            f"MERGE (e1)-[:FOLLOWS]->(e2);")
        
        lines.append("")
        lines.append("// ============================================================")
        lines.append("// CAUSAL EDGES - Event → Event")
        lines.append("// ============================================================\n")
        
        # 14. Event -[:CAUSES]-> Event
        lines.append("// 14. EVENT CAUSES EVENT")
        for link in self.causal_links:
            escaped_mechanism = _truncate_safe(_escape_cypher_string(link.mechanism), 200)
            lines.append(f"MATCH (cause:Event {{id: \"{link.cause_id}\"}}), "
                        f"(effect:Event {{id: \"{link.effect_id}\"}}) "
                        f"MERGE (cause)-[:CAUSES {{"
                        f"relationType: \"{link.relationType}\", "
                        f"mechanism: \"{escaped_mechanism}\", "
                        f"sign: \"{link.sign}\", "
                        f"weight: {link.weight}, "
                        f"confidence: {link.confidence}, "
                        f"cause_seq: {link.cause_sequence}, "
                        f"effect_seq: {link.effect_sequence}"
                        f"}}]->(effect);")
        
        lines.append("")
        lines.append("// ============================================================")
        lines.append("// DUAL FLOW HIERARCHY COMPLETE")
        lines.append(f"// Total Events: {len(self.events)}")
        lines.append(f"// Total Event→Entity productions: {len(self.event_produces)}")
        lines.append(f"// Total Entity→Event links: {len(self.entity_points_to)}")
        lines.append(f"// Total Causal Links: {len(self.causal_links)}")
        lines.append("// Flow: Event → Entity (production)")
        lines.append("//       Entity → NextEvent (continuation)")
        lines.append("//       Event → Event (causation)")
        lines.append("// ============================================================")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write('\n'.join(lines))
        
        print(f"[export] Cypher exported: {len(lines)} statements")

    def export_csv(self, out_dir: str = "neo4j_csv") -> Dict[str, str]:
        """Export DUAL FLOW structure to Neo4j CSV format"""
        import csv
        
        os.makedirs(out_dir, exist_ok=True)
        
        # Collect unique entities
        entities_by_type = defaultdict(dict)
        for prod in self.event_produces:
            entities_by_type[prod.entity_type][prod.entity_id] = prod.entity_name
        
        # Event nodes
        events_rows = []
        for ev in self.events:
            events_rows.append({
                ":ID": ev.id,
                "name": ev.name,
                "eventType": ev.eventType,
                "actionType": ev.actionType,
                "source_quote": ev.source_quote,
                "causeWeight": ev.causeWeight or 0.0,
                "confidence": ev.confidence,
                "sequence": ev.sequence,
                "chapter": ev.chapter,
                "time": ev.time or "",
                "location": ev.location or ""
            })
        
        # Agent nodes
        all_agents = {}
        all_agents.update(entities_by_type.get("actor", {}))
        all_agents.update(entities_by_type.get("patient", {}))
        agent_rows = [{":ID": aid, "name": name} for aid, name in all_agents.items()]
        
        # Place nodes
        place_rows = [{":ID": pid, "name": name} for pid, name in entities_by_type.get("place", {}).items()]
        
        # WhyFactor nodes
        whyfactor_rows = [{":ID": wid, "factor": name} for wid, name in entities_by_type.get("whyfactor", {}).items()]
        
        # Event → Entity edges
        produces_actor_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_ACTOR",
            "strength": prod.strength
        } for prod in self.event_produces if prod.entity_type == "actor"]
        
        produces_patient_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_PATIENT",
            "strength": prod.strength
        } for prod in self.event_produces if prod.entity_type == "patient"]
        
        produces_motivation_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_MOTIVATION",
            "weight": prod.strength
        } for prod in self.event_produces if prod.entity_type == "whyfactor"]
        
        produces_location_rows = [{
            ":START_ID": prod.event_id,
            ":END_ID": prod.entity_id,
            ":TYPE": "PRODUCES_LOCATION",
            "specificity": prod.strength
        } for prod in self.event_produces if prod.entity_type == "place"]
        
        # Entity → Event edges
        acts_in_rows = [{
            ":START_ID": ept.entity_id,
            ":END_ID": ept.next_event_id,
            ":TYPE": "ACTS_IN",
            "strength": ept.strength
        } for ept in self.entity_points_to if ept.relationship == "ACTS_IN"]
        
        affected_in_rows = [{
            ":START_ID": ept.entity_id,
            ":END_ID": ept.next_event_id,
            ":TYPE": "AFFECTED_IN",
            "strength": ept.strength
        } for ept in self.entity_points_to if ept.relationship == "AFFECTED_IN"]
        
        motivates_rows = [{
            ":START_ID": ept.entity_id,
            ":END_ID": ept.next_event_id,
            ":TYPE": "MOTIVATES",
            "weight": ept.strength
        } for ept in self.entity_points_to if ept.relationship == "MOTIVATES"]
        
        hosts_rows = [{
            ":START_ID": ept.entity_id,
            ":END_ID": ept.next_event_id,
            ":TYPE": "HOSTS",
            "specificity": ept.strength
        } for ept in self.entity_points_to if ept.relationship == "HOSTS"]
        
        # Event -[:FOLLOWS]-> Event
        follows_rows = []
        for i in range(len(self.events) - 1):
            ev1 = self.events[i]
            ev2 = self.events[i + 1]
            if ev1.chapter == ev2.chapter:
                follows_rows.append({
                    ":START_ID": ev1.id,
                    ":END_ID": ev2.id,
                    ":TYPE": "FOLLOWS"
                })
        
        # Event -[:CAUSES]-> Event
        causes_rows = [{
            ":START_ID": link.cause_id,
            ":END_ID": link.effect_id,
            ":TYPE": "CAUSES",
            "relationType": link.relationType,
            "mechanism": link.mechanism,
            "sign": link.sign,
            "weight": link.weight,
            "confidence": link.confidence,
            "cause_seq": link.cause_sequence,
            "effect_seq": link.effect_sequence
        } for link in self.causal_links]

        files = {
            "events.csv": events_rows,
            "agents.csv": agent_rows,
            "places.csv": place_rows,
            "whyfactors.csv": whyfactor_rows,
            "produces_actor.csv": produces_actor_rows,
            "produces_patient.csv": produces_patient_rows,
            "produces_motivation.csv": produces_motivation_rows,
            "produces_location.csv": produces_location_rows,
            "acts_in.csv": acts_in_rows,
            "affected_in.csv": affected_in_rows,
            "motivates.csv": motivates_rows,
            "hosts.csv": hosts_rows,
            "follows.csv": follows_rows,
            "causes.csv": causes_rows,
        }

        def _write_csv(rows, path):
            if not rows:
                open(path, "w", encoding="utf-8").close()
                return
            
            if pd is not None:
                pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
            else:
                keys = list(rows[0].keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(rows)

        out_paths = {}
        for fname, rows in files.items():
            path = os.path.join(out_dir, fname)
            _write_csv(rows, path)
            out_paths[fname] = path
        
        return out_paths

    async def run_async(self, text_path: str, 
                       out_json: str = "ge_preprocessed.json",
                       out_cypher: str = "ge_import.cypher",
                       out_csv_dir: str = "neo4j_csv",
                       max_chapters: Optional[int] = None,
                       batch_size: int = BATCH_SIZE,
                       causal_window: int = 4,
                       causal_sample_rate: float = SAMPLE_RATE,
                       causal_batch_size: int = CAUSAL_BATCH_SIZE) -> Dict[str, Any]:
        """Main async pipeline with DUAL FLOW"""
        
        print("[pipeline] Loading text...")
        raw = self.load_text(text_path)
        chapters = self.split_chapters(raw)

        if max_chapters is not None:
            chapters = chapters[:max_chapters]

        print(f"[pipeline] Processing {len(chapters)} chapters")
        print(f"[pipeline] DUAL FLOW: Event→Entity→Event chains ✓")

        # Process each chapter
        for chapter_id, chapter_text in chapters:
            try:
                events, produces = await self.process_chapter_events(
                    chapter_id, chapter_text, batch_size
                )
                self.events.extend(events)
                self.event_produces.extend(produces)
                print(f"[chapter {chapter_id}] Total events: {len(self.events)}, "
                      f"entity productions: {len(self.event_produces)}")
            except Exception as e:
                print(f"[error] Chapter {chapter_id} failed: {e}")
                traceback.print_exc()
                continue

        if not self.events:
            raise CEKGError("No events were extracted")

        # --- NEW STEPS ---
        # Sort all events by sequence *before* propagating context
        print("[pipeline] Sorting all events by sequence...")
        self.events.sort(key=lambda x: (x.chapter, x.sequence))
        # Re-assign sequence numbers to be globally unique and ordered
        for i, event in enumerate(self.events):
            event.sequence = i

        # Run Pass 2: Propagate context
        self.propagate_context()
        # --- END NEW STEPS ---

        # Create entity→event links (now using the propagated data)
        self.create_entity_to_event_links()

        # Causal linking
        print(f"[pipeline] Starting causal linking")
        try:
            await self._batch_causal_linking(
                window=causal_window,
                sample_rate=causal_sample_rate,
                batch_size=causal_batch_size
            )
        except Exception as e:
            print(f"[warning] Causal linking failed: {e}")
            traceback.print_exc()
        
        # Export results
        print(f"[pipeline] Exporting results...")
        try:
            self.export_json(out_json)
            self.export_neo4j_cypher(out_cypher)
            csvs = self.export_csv(out_csv_dir)
        except Exception as e:
            print(f"[error] Export failed: {e}")
            traceback.print_exc()
            raise

        dag_stats = self.dag_validator.get_stats()

        return {
            "json": out_json, 
            "cypher": out_cypher, 
            "csv": csvs,
            "stats": {
                "events": len(self.events),
                "event_produces_entity": len(self.event_produces),
                "entity_points_to_event": len(self.entity_points_to),
                "causal_links": len(self.causal_links),
                "cached_event_extractions": await event_extraction_cache.size(),
                "cached_causal_assessments": await assessment_cache.size(),
                "chapters_processed": len(set(ev.chapter for ev in self.events)),
                "dag_stats": dag_stats
            }
        }

    def run(self, *args, **kwargs):
        """Synchronous wrapper for async run"""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call run() from within an event loop. "
                "Use 'await run_async()' instead."
            )
        except RuntimeError as e:
            if "cannot call run()" in str(e).lower():
                raise
            return asyncio.run(self.run_async(*args, **kwargs))


# ----------------------------- CLI entrypoint -------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CEKG Preprocessor (DUAL FLOW)")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--out-json", default="ge_preprocessed.json")
    parser.add_argument("--out-cypher", default="ge_import.cypher")
    parser.add_argument("--out-csv", default="neo4j_csv")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--max-chapters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--causal-batch-size", type=int, default=CAUSAL_BATCH_SIZE)
    parser.add_argument("--causal-window", type=int, default=4)
    parser.add_argument("--causal-sample-rate", type=float, default=SAMPLE_RATE)

    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='sk-...'")
        exit(1)

    try:
        preprocessor = CEKGPreprocessor(openai_model=args.openai_model)
        print(f"[config] model: {preprocessor.openai_model}")
        print(f"[config] batch_size: {args.batch_size}, causal_batch_size: {args.causal_batch_size}")
        print(f"[config] cache_max_size: {CACHE_MAX_SIZE}")
        print(f"[config] DUAL FLOW: Event→Entity→Event chains ✓")
        
        start_time = time.time()
        out = preprocessor.run(
            text_path=args.input, 
            out_json=args.out_json, 
            out_cypher=args.out_cypher, 
            out_csv_dir=args.out_csv,
            max_chapters=args.max_chapters,
            batch_size=args.batch_size,
            causal_window=args.causal_window,
            causal_sample_rate=args.causal_sample_rate,
            causal_batch_size=args.causal_batch_size
        )
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✓ Finished in {elapsed:.2f} seconds")
        print(f"✓ Events: {out['stats']['events']}")
        print(f"✓ Event→Entity productions: {out['stats']['event_produces_entity']}")
        print(f"✓ Entity→Event links: {out['stats']['entity_points_to_event']}")
        print(f"✓ Causal links (Event→Event): {out['stats']['causal_links']}")
        print(f"✓ Cached event extractions: {out['stats']['cached_event_extractions']}")
        print(f"✓ Cached causal assessments: {out['stats']['cached_causal_assessments']}")
        print(f"✓ Chapters processed: {out['stats']['chapters_processed']}")
        
        dag = out['stats']['dag_stats']
        print(f"✓ DAG validated: {dag['nodes']} nodes, {dag['edges']} edges")
        print(f"  └─ Max in-degree: {dag['max_in_degree']}, Max out-degree: {dag['max_out_degree']}")
        
        print(f"✓ Outputs: {out['json']}, {out['cypher']}")
        print(f"✓ CSV files: {len(out['csv'])} files in {args.out_csv}/")
        print(f"\n{'='*60}")
        print("DUAL FLOW ARCHITECTURE:")
        print("  Event₁ -[:PRODUCES_ACTOR]→ Agent")
        print("         Agent -[:ACTS_IN]→ Event₂")
        print("  Event₁ -[:PRODUCES_PATIENT]→ Agent")
        print("         Agent -[:AFFECTED_IN]→ Event₂")
        print("  Event₁ -[:PRODUCES_MOTIVATION]→ WhyFactor")
        print("         WhyFactor -[:MOTIVATES]→ Event₂")
        print("  Event₁ -[:PRODUCES_LOCATION]→ Place")
        print("         Place -[:HOSTS]→ Event₂")
        print("  Event -[:CAUSES]→ Event (primary causal hierarchy)")
        print("")
        print("Chain structure: Event₁ → Entity → Event₂ → Entity → Event₃")
        print("Each entity instance points to its NEXT occurrence")
        print("Initial entities naturally exist before first event")
        print("No bidirectional edges - strict directional flow")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ FATAL ERROR: {e}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        exit(1)