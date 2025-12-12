"""
Enhanced LLM Service with Agent Type Classification
"""
import json
import asyncio
import re
from typing import List, Dict, Optional, Any

try:
    import openai
except ImportError:
    openai = None

# Assumed internal imports based on context
from .schemas import ExtractionError, CEKEvent
from .utils import BoundedCache, _hash_for_cache
from .config import CACHE_MAX_SIZE

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
event_extraction_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
assessment_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
semantic_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
scene_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
agent_classification_cache = BoundedCache(max_size=CACHE_MAX_SIZE)

# ---------------------------------------------------------------------------
# Default Ontologies
# ---------------------------------------------------------------------------
DEFAULT_EVENT_ONTOLOGY = [
    "PHYSICAL_MOVEMENT", "COMMUNICATION_VERBAL", "INTERNAL_THOUGHT",
    "EMOTIONAL_REACTION", "OBSERVATION", "CONFLICT_PHYSICAL",
    "SOCIAL_INTERACTION", "STATE_CHANGE", "ACQUISITION", "TRAVEL"
]

DEFAULT_RELATIONSHIP_ONTOLOGY = [
    "DIRECT_CAUSE", "ENABLES", "PREVENTS", "TRIGGERS", "MOTIVATES",
    "INTERRUPTS", "INHIBITS", "PRECEDES"
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
JSON_START = "json"
JSON_END = ""

PROMPT_DETAILED_INSTRUCTIONS = """Extract EVERY distinct event from this text. An event is ANY action, perception, dialogue, emotion, or state change. Be granular."""
PROMPT_HIGH_LEVEL_INSTRUCTIONS = """Extract only the MAJOR narrative events or ideas from this text."""
PROMPT_EXPANSION_ADDON = """
Additional Tasks:
- Find Implicit Events (psychological/emotional).
- Group Compound Events.
- Coreference Resolution (Normalize names).
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
1. **raw_description**: The natural language description of the event.
2. **event_category**: Choose the BEST fit from the Provided Ontology List below.
3. **location_context**: The setting. If implied, you may leave null.
4. **time_context**: Temporal marker (e.g., "Morning").
5. **actors/patients**: CRITICAL - Use FULL PROPER NAMES ONLY:
   - ✓ CORRECT: "Philip Pirrip", "Abel Magwitch", "Mrs. Joe Gargery"
   - ✗ WRONG: "he", "him", "the boy", "the convict", "narrator"
   - If you see a pronoun (he/she/they/it), resolve it to the character's name
   - If you see a descriptor (the boy/the man), identify WHO it refers to
   - If the character's full name is unknown, use the name they're called in text
   - DO NOT include: pronouns, "narrator", "speaker", generic descriptors like "man"/"woman"
   - Only include characters WHO PERFORM OR RECEIVE the action, not mentioned characters
6. **why_factors**: List of strings explaining immediate causes.

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
Input JSON: {event_list_json}
Output JSON: {{"scenes": [{{"event_ids": [], "theme": "...", "confidence": 0.9}}]}}"""

PROMPT_AGENT_CLASSIFICATION = """You are a narrative theory expert.

Analyze this character's role in the story and classify them using ONE agent type from the list below.

**Character Name:** {character_name}

**Their Actions in the Story:**
{event_descriptions}

**Available Agent Types:**
{agent_type_list}

**Instructions:**
1. Consider the character's NARRATIVE FUNCTION, not just their actions
2. Choose the SINGLE BEST agent type that describes their primary role
3. Return ONLY valid JSON with these fields:
   - agentType: ONE type from the list above
   - explanation: Brief justification (max 30 words)
   - confidence: float 0.0-1.0

Return ONLY the JSON object:"""

# ---------------------------------------------------------------------------
# Service Functions
# ---------------------------------------------------------------------------

def init_openai_client(api_key: str) -> Any:
    if openai is None:
        raise RuntimeError("openai package not installed.")
    return openai.OpenAI(api_key=api_key)

async def _async_llm_json_call(prompt: str, model: str, client: Any, 
                               cache: BoundedCache, cache_key: str, 
                               max_tokens: int = 4096) -> Any:
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached, None

    for attempt in range(3):
        try:
            loop = asyncio.get_event_loop()
            
            def make_req():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=0.0,
                    timeout=30
                )
            
            resp = await loop.run_in_executor(None, make_req)
            text = resp.choices[0].message.content.strip()
            
            json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                cleaned = ''.join([c for c in text if ord(c) >= 32 or c in '\n\r\t'])
                data = json.loads(cleaned)
            
            if isinstance(data, dict):
                if 'events' in data and isinstance(data['events'], list):
                    data = data['events']
                elif 'scenes' in data and isinstance(data['scenes'], list):
                    data = data['scenes']

            await cache.set(cache_key, data)
            return data, getattr(resp.choices[0], "logprobs", None)
            
        except Exception as e:
            if attempt == 2:
                print(f"[error] LLM call failed after 3 attempts: {e}")
                return [], None
            await asyncio.sleep(1)
    return [], None

def _get_extraction_prompt(text_input, chapter_id, extraction_style, 
                           enable_llm_expansion, event_ontology):
    instr = PROMPT_HIGH_LEVEL_INSTRUCTIONS if extraction_style == "high-level" else PROMPT_DETAILED_INSTRUCTIONS
    exp = PROMPT_EXPANSION_ADDON if enable_llm_expansion else ""
    if not event_ontology:
        event_ontology = DEFAULT_EVENT_ONTOLOGY
    return PROMPT_BASE_TEMPLATE.format(
        instructions=instr, expansion_instructions=exp,
        chapter_id=chapter_id, text_input=text_input,
        ontology_list=", ".join(event_ontology)
    )

async def extract_events_from_text(text_input, chapter_id, model, client,
                                   enable_llm_expansion, request_logprobs,
                                   extraction_style, event_ontology=None):
    ont_hash = str(hash(tuple(event_ontology))) if event_ontology else "default"
    key = _hash_for_cache(
        f"{chapter_id}:{text_input}:{extraction_style}:{enable_llm_expansion}:{ont_hash}:v3",
        model
    )
    prompt = _get_extraction_prompt(
        text_input.strip(), chapter_id, extraction_style,
        enable_llm_expansion, event_ontology
    )
    data, logprobs = await _async_llm_json_call(
        prompt, model, client, event_extraction_cache, key, 16000
    )
    return (data if isinstance(data, list) else [data]), logprobs

async def batch_extract_events(paragraphs, model, client, enable_llm_expansion,
                               request_logprobs, extraction_style, event_ontology=None):
    tasks = [
        extract_events_from_text(p, cid, model, client, enable_llm_expansion,
                                 request_logprobs, extraction_style, event_ontology)
        for p, cid in paragraphs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [(r if not isinstance(r, Exception) else ([], None)) for r in results]

async def batch_assess_pairs(pairs, model, client, relationship_ontology=None):
    if not relationship_ontology:
        relationship_ontology = DEFAULT_RELATIONSHIP_ONTOLOGY
    ontology_str = ", ".join(relationship_ontology)
    ont_hash = str(hash(tuple(relationship_ontology)))
    
    async def _assess(cause, effect):
        key = _hash_for_cache(f"{cause}|||{effect}|||{ont_hash}", model)
        prompt = PROMPT_CAUSAL_PAIR.format(
            cause_text=cause, effect_text=effect, relation_ontology=ontology_str
        )
        try:
            data, _ = await _async_llm_json_call(
                prompt, model, client, assessment_cache, key, 1024
            )
            return data
        except:
            return None
    
    tasks = [_assess(c, e) for c, e, _, _ in pairs]
    return await asyncio.gather(*tasks)

async def batch_assess_semantic_pairs(pairs, model, client):
    async def _assess_sem(cause, effect):
        key = _hash_for_cache(f"sem:{cause}|||{effect}", model)
        prompt = PROMPT_SEMANTIC_PAIR.format(cause_text=cause, effect_text=effect)
        try:
            data, _ = await _async_llm_json_call(
                prompt, model, client, semantic_cache, key, 1024
            )
            return data
        except:
            return None
    
    tasks = [_assess_sem(c, e) for c, e, _, _ in pairs]
    return await asyncio.gather(*tasks)

async def extract_scenes_from_chapter_async(chapter_events, chapter_id, model, client):
    if len(chapter_events) > 300:
        chapter_events = chapter_events[:300]
    simple = [{"id": e.id, "desc": e.raw_description, "seq": e.sequence} 
              for e in chapter_events]
    prompt = PROMPT_SCENE_GROUPING.format(event_list_json=json.dumps(simple))
    key = _hash_for_cache(f"scene:{chapter_id}:{len(simple)}", model)
    try:
        data, _ = await _async_llm_json_call(
            prompt, model, client, scene_cache, key, 8192
        )
        return data if isinstance(data, list) else []
    except:
        return []

async def classify_agent_type(character_name: str, event_descriptions: List[str],
                              agent_type_names: List[str], model: str, 
                              client: Any) -> str:
    """
    NEW: Classify a character's agent type using LLM.
    Returns the agent type name (e.g., "PROTAGONIST_HERO")
    """
    # Truncate event list if too long
    events_text = "\n".join([f"- {desc[:100]}" for desc in event_descriptions[:15]])
    types_text = "\n".join([f"- {name}" for name in agent_type_names])
    
    prompt = PROMPT_AGENT_CLASSIFICATION.format(
        character_name=character_name,
        event_descriptions=events_text,
        agent_type_list=types_text
    )
    
    key = _hash_for_cache(f"agent:{character_name}:{len(event_descriptions)}", model)
    
    try:
        data, _ = await _async_llm_json_call(
            prompt, model, client, agent_classification_cache, key, 512
        )
        
        if isinstance(data, dict) and "agentType" in data:
            return data["agentType"]
        return "STRUCTURAL_AGENT"  # Default fallback
    except Exception as e:
        print(f"[warning] Agent classification failed for {character_name}: {e}")
        return "STRUCTURAL_AGENT"

async def get_cache_sizes():
    return {
        "event_extraction_cache_size": await event_extraction_cache.size(),
        "assessment_cache_size": await assessment_cache.size(),
        "semantic_cache_size": await semantic_cache.size(),
        "scene_cache_size": await scene_cache.size(),
        "agent_classification_cache_size": await agent_classification_cache.size()
    }