import json
import asyncio
import re
from typing import List, Dict, Optional, Any

try:
    import openai
except ImportError:
    openai = None

from .schemas import ExtractionError, CEKEvent
from .utils import BoundedCache, _hash_for_cache
from .config import CACHE_MAX_SIZE

# --- Caches ---
event_extraction_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
assessment_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
semantic_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
scene_cache = BoundedCache(max_size=CACHE_MAX_SIZE)

# --- DEFAULTS (Fallbacks if no dictionary file is found) ---
DEFAULT_EVENT_ONTOLOGY = [
    "PHYSICAL_MOVEMENT", "COMMUNICATION_VERBAL", "INTERNAL_THOUGHT", 
    "EMOTIONAL_REACTION", "OBSERVATION", "CONFLICT_PHYSICAL", 
    "SOCIAL_INTERACTION", "STATE_CHANGE", "ACQUISITION", "TRAVEL"
]

DEFAULT_RELATIONSHIP_ONTOLOGY = [
    "DIRECT_CAUSE", "ENABLES", "PREVENTS", "TRIGGERS", "MOTIVATES", 
    "INTERRUPTS", "INHIBITS", "PRECEDES"
]

# --- PROMPTS ---

JSON_START = "```" + "json"
JSON_END = "```"

PROMPT_DETAILED_INSTRUCTIONS = """Extract EVERY distinct event from this text. An event is ANY action, perception, dialogue, emotion, or state change. Be granular."""
PROMPT_HIGH_LEVEL_INSTRUCTIONS = """Extract only the MAJOR narrative events or ideas from this text. Focus on the main plot points."""

PROMPT_EXPANSION_ADDON = """
**Additional Tasks:**
1. Find Implicit Events (psychological/emotional).
2. Group Compound Events (e.g., 'walked and knocked' -> "knock on door").
3. Coreference Resolution (Normalize names to full canonical name).
"""

PROMPT_BASE_TEMPLATE = f"""You are an expert literary analyst.

**Your Goal:**
Extract narrative events from the text below.
{{instructions}}
{{expansion_instructions}}

**Output Format:**
You MUST return a valid JSON object wrapped in a markdown code block.
Example:
{JSON_START}
{{{{
  "events": [
    {{{{
      "raw_description": "Pip ran towards the church door",
      "event_category": "PHYSICAL_MOVEMENT", 
      "actors": ["Pip"],
      "patients": [],
      "location_context": "Churchyard",
      "time_context": "Late afternoon",
      "why_factors": ["Fear of Magwitch"],
      "confidence": 0.95
    }}}}
  ]
}}}}
{JSON_END}

**Fields Guide:**
1. **raw_description**: The natural language description of the event (summarized from text).
2. **event_category**: Choose the BEST fit from the Provided Ontology List below.
3. **location_context**: The setting. If implied by previous context, you may leave it null or fill it if obvious.
4. **time_context**: Temporal marker (e.g., "Morning", "After dinner").
5. **actors/patients**: Must be SENTIENT beings (Characters).
6. **why_factors**: List of strings explaining the immediate cause/motivation.

**Provided Ontology List (event_category):**
[{{ontology_list}}]

**Text to Analyze (Chapter {{chapter_id}}):**
{{text_input}}

Return ONLY the JSON object within the code block:"""

PROMPT_CAUSAL_PAIR = (
    "You are a literary causal analyst.\n"
    "Determine the precise relationship between Event A and Event B using the provided dictionary.\n\n"
    "**Relationship Dictionary:**\n"
    "[{relation_ontology}]\n\n"
    "**Output Format:**\n"
    "Return a JSON object:\n"
    "- relationType: ONE valid category from the dictionary above (or \"NONE\")\n"
    "- mechanism: short explanation (<=20 words)\n"
    "- weight: float 0.0-1.0\n"
    "- confidence: float 0.0-1.0\n\n"
    "EVENT A (Cause/Prior): \"{cause_text}\"\n"
    "EVENT B (Effect/Posterior): \"{effect_text}\"\n\n"
    "Return ONLY the JSON object:"
)

PROMPT_SEMANTIC_PAIR = (
    "You are a literary analyst.\n"
    "Determine if there is a **non-causal semantic link** (explanation, contrast).\n"
    "Return JSON:\n"
    "- relation: [\"explanation\", \"contrast\", \"none\"]\n"
    "- cue: list of keywords\n"
    "- confidence: float\n\n"
    "EVENT 1: \"{cause_text}\"\n"
    "EVENT 2: \"{effect_text}\"\n\n"
    "Return ONLY the JSON object:"
)

PROMPT_SCENE_GROUPING = """Group these events into narrative scenes.
A scene is a cluster of events in the same time/location/theme.

Input JSON: {event_list_json}

Output JSON format:
{{
  "scenes": [
    {{
      "event_ids": ["id1", "id2"],
      "theme": "Theme Name",
      "confidence": 0.9
    }}
  ]
}}
"""

# --- Functions ---

def init_openai_client(api_key: str) -> openai.OpenAI:
    if openai is None:
        raise RuntimeError("openai package not installed.")
    return openai.OpenAI(api_key=api_key)

async def _async_llm_json_call(
    prompt: str, 
    model: str, 
    client: openai.OpenAI, 
    cache: BoundedCache, 
    cache_key: str,
    max_tokens: int = 4096
) -> Any:
    
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached, None 

    for attempt in range(3):
        try:
            loop = asyncio.get_event_loop()
            
            def make_request():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=0.0
                )
            
            resp = await loop.run_in_executor(None, make_request)
            text = resp.choices[0].message.content.strip()
            
            # Markdown Extraction
            json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()
            
            # JSON Parse
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Simple repair for common trailing comma or bracket issues
                cleaned_chars = [c for c in text if ord(c) >= 32 or c in '\n\r\t']
                text = ''.join(cleaned_chars)
                data = json.loads(text)
            
            # Unwrap common wrappers
            if isinstance(data, dict):
                if 'events' in data and isinstance(data['events'], list):
                    data = data['events']
                elif 'scenes' in data and isinstance(data['scenes'], list):
                    data = data['scenes']

            await cache.set(cache_key, data)
            return data, getattr(resp.choices[0], "logprobs", None)
            
        except Exception as e:
            if attempt == 2:
                # raise ExtractionError(f"LLM call failed: {e}")
                print(f"[error] LLM failed after 3 attempts: {e}")
                return [], None
            await asyncio.sleep(1)
    return [], None


def _get_extraction_prompt(
    text_input: str,
    chapter_id: int,
    extraction_style: str,
    enable_llm_expansion: bool,
    event_ontology: List[str]
) -> str:
    if extraction_style == "high-level":
        instructions = PROMPT_HIGH_LEVEL_INSTRUCTIONS
    else: 
        instructions = PROMPT_DETAILED_INSTRUCTIONS
    
    expansion_instructions = PROMPT_EXPANSION_ADDON if enable_llm_expansion else ""
    
    # Inject Dictionary
    if not event_ontology:
        event_ontology = DEFAULT_EVENT_ONTOLOGY
    ontology_str = ", ".join(event_ontology)
        
    return PROMPT_BASE_TEMPLATE.format(
        instructions=instructions,
        expansion_instructions=expansion_instructions,
        chapter_id=chapter_id, 
        text_input=text_input,
        ontology_list=ontology_str
    )

async def extract_events_from_text(
    text_input: str, 
    chapter_id: int, 
    model: str,
    client: openai.OpenAI,
    enable_llm_expansion: bool,
    request_logprobs: bool,
    extraction_style: str,
    event_ontology: List[str] = None
) -> (Any, Any):
    
    # Include ontology hash in cache key so changing dictionary refreshes cache
    ont_hash = str(hash(tuple(event_ontology))) if event_ontology else "default"
    
    cache_key = _hash_for_cache(
        f"{chapter_id}:{text_input}:{extraction_style}:{enable_llm_expansion}:{ont_hash}:v3", 
        model
    )
    
    prompt = _get_extraction_prompt(
        text_input=text_input.strip(),
        chapter_id=chapter_id,
        extraction_style=extraction_style,
        enable_llm_expansion=enable_llm_expansion,
        event_ontology=event_ontology
    )

    data, logprobs = await _async_llm_json_call(
        prompt=prompt,
        model=model,
        client=client,
        cache=event_extraction_cache,
        cache_key=cache_key,
        max_tokens=16000
    )
    
    if not isinstance(data, list):
        data = [data]
    
    return data, logprobs

async def batch_extract_events(
    paragraphs: List[tuple[str, int]], 
    model: str,
    client: openai.OpenAI,
    enable_llm_expansion: bool,
    request_logprobs: bool,
    extraction_style: str,
    event_ontology: List[str] = None
) -> List[tuple[Any, Any]]:
    
    tasks = [
        extract_events_from_text(
            para, chapter_id, model, client, 
            enable_llm_expansion, request_logprobs, extraction_style, event_ontology
        )
        for para, chapter_id in paragraphs
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[batch_extract] Paragraph {i} failed: {result}")
            processed_results.append(([], None))
        else:
            processed_results.append(result)
    
    return processed_results

async def batch_assess_pairs(
    pairs: List[tuple[str, str, str, str]], 
    model: str,
    client: openai.OpenAI,
    relationship_ontology: List[str] = None
) -> List[Optional[Dict[str, Any]]]:
    
    if not relationship_ontology:
        relationship_ontology = DEFAULT_RELATIONSHIP_ONTOLOGY
    
    ontology_str = ", ".join(relationship_ontology)
    ont_hash = str(hash(tuple(relationship_ontology)))

    async def _assess(cause_text, effect_text):
        cache_key = _hash_for_cache(f"{cause_text}|||{effect_text}|||{ont_hash}", model)
        prompt = PROMPT_CAUSAL_PAIR.format(
            cause_text=cause_text, 
            effect_text=effect_text,
            relation_ontology=ontology_str
        )

        try:
            data, _ = await _async_llm_json_call(
                prompt=prompt, model=model, client=client,
                cache=assessment_cache, cache_key=cache_key,
                max_tokens=1024 
            )
            return data
        except Exception as e:
            return None

    tasks = [
        _assess(cause_text, effect_text)
        for cause_text, effect_text, _, _ in pairs
    ]
    return await asyncio.gather(*tasks)

async def batch_assess_semantic_pairs(
    pairs: List[tuple[str, str, str, str]], 
    model: str,
    client: openai.OpenAI
) -> List[Optional[Dict[str, Any]]]:
    
    async def _assess_sem(cause, effect):
        cache_key = _hash_for_cache(f"sem:{cause}|||{effect}", model)
        prompt = PROMPT_SEMANTIC_PAIR.format(cause_text=cause, effect_text=effect)
        try:
            data, _ = await _async_llm_json_call(
                prompt=prompt, model=model, client=client,
                cache=semantic_cache, cache_key=cache_key, max_tokens=1024
            )
            return data
        except: return None

    tasks = [_assess_sem(c, e) for c, e, _, _ in pairs]
    return await asyncio.gather(*tasks)

async def extract_scenes_from_chapter_async(
    chapter_events: List[CEKEvent],
    chapter_id: int,
    model: str,
    client: openai.OpenAI
) -> List[Dict[str, Any]]:
    
    if len(chapter_events) > 300:
         chapter_events = chapter_events[:300]

    event_list_simple = [
        {"id": ev.id, "desc": ev.raw_description, "seq": ev.sequence} 
        for ev in chapter_events
    ]
    event_list_json = json.dumps(event_list_simple, indent=4)
    
    cache_key = _hash_for_cache(event_list_json, model)
    prompt = PROMPT_SCENE_GROUPING.format(
        chapter_id=chapter_id,
        event_list_json=event_list_json
    )
    
    try:
        data, _ = await _async_llm_json_call(
            prompt=prompt, model=model, client=client,
            cache=scene_cache, cache_key=cache_key,
            max_tokens=8192
        )
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[error] Scene grouping for chapter {chapter_id} failed: {e}")
        return []

async def get_cache_sizes() -> dict:
    return {
        "event_extraction_cache_size": await event_extraction_cache.size(),
        "assessment_cache_size": await assessment_cache.size(),
        "semantic_cache_size": await semantic_cache.size(),
        "scene_cache_size": await scene_cache.size()
    }