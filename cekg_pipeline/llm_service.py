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

# --- Caches are now local to this service ---
event_extraction_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
assessment_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
semantic_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
scene_cache = BoundedCache(max_size=CACHE_MAX_SIZE)

# --- PROMPT SET 1: INSTRUCTIONS (THE GOAL) ---

PROMPT_DETAILED_INSTRUCTIONS = """Extract EVERY distinct event from this text. An event is ANY action, perception, dialogue, emotion, or state change. Be granular. Capturing specific sequences of actions is important."""

PROMPT_HIGH_LEVEL_INSTRUCTIONS = """Extract only the MAJOR narrative events or ideas from this text. Do NOT extract every single verb or minor action. Focus on the main plot points, significant changes in state, or the core "idea-to-idea" flow."""

# --- PROMPT SET 2: EXPANSION (THE MODIFIER) ---

PROMPT_EXPANSION_ADDON = """
**Additional Tasks:**
1.  **Find Implicit Events:** Extract psychological or emotional events (e.g., 'Pip felt ashamed').
2.  **Group Compound Events:** Combine simple, sequential actions into one event (e.g., 'walked to the door and knocked' -> "knock on door").
3.  **Coreference Resolution:** Normalize all actor/patient names to their **full, canonical name** (e.g., "the boy", "Philip Pirrip", "Pip" -> "Pip").
"""

# --- PROMPT SET 3: THE BASE TEMPLATE ---

# UPDATED: Added strict constraints to 'actors' and 'patients' to exclude body parts/objects.
PROMPT_BASE_TEMPLATE = """You are an expert literary analyst.

**Your Goal:**
{instructions}
{expansion_instructions}

**Output Format:**
You MUST return a valid JSON object wrapped in a markdown code block.
Example:
""" + "```json" + """
{{
  "events": [...]
}}
""" + "```" + """

Each event object in the "events" list must have:
- name: brief description
- eventType: one of ["perception/recognition", "communication", "conflict", "assistance", "movement", "emotion", "transformation", "other"]
- actionType: canonical verb
- actors: array of {{"name": "Name", "strength": 0.0-1.0}}. IMPORTANT: Actors must be SENTIENT beings (people, animals, personified entities). Do NOT extract body parts (e.g., "liver", "heart"), objects, or abstract concepts as actors.
- patients: array of {{"name": "Name", "strength": 0.0-1.0}}. IMPORTANT: Patients must be SENTIENT beings. If an object/body part is affected, leave this empty or name the owner of the part.
- location: The specific place mentioned.
- time: temporal reference.
- whyFactors: **Mandatory** array of {{"factor": "desc", "weight": 0.0-1.0, "category": "Immediate/Contextual"}}.
- confidence: 0.0-1.0 (your confidence in the extraction)
- quote: relevant text (max 50 words).

**Text to Analyze (Chapter {chapter_id}):**
{text_input}

Return ONLY the JSON object within the code block:"""

# --- PROMPT SET 4: LINKING & SCENES ---

PROMPT_CAUSAL_PAIR = (
    "You are a careful literary causal analyst.\n"
    "Given two event descriptions, return a single JSON object with:\n"
    "- relationType: one of [\"causes\", \"enables\", \"prevents\", \"aggravates\", \"none\"]\n"
    "- mechanism: short explanation (<=50 words)\n"
    "- sign: + or - or 0\n"
    "- weight: float 0.0-1.0\n"
    "- confidence: float 0.0-1.0\n\n"
    "CAUSE: \"{cause_text}\"\n"
    "EFFECT: \"{effect_text}\"\n\n"
    "Return ONLY the JSON object:"
)

PROMPT_SEMANTIC_PAIR = (
    "You are a literary analyst focusing on narrative structure.\n"
    "Given two events, determine if there is a **non-causal semantic link** between them.\n"
    "A semantic link is for **explanation, elaboration, contrast, or parallelism**.\n"
    "Return a single JSON object with:\n"
    "- relation: one of [\"explanation\", \"elaboration\", \"contrast\", \"parallelism\", \"none\"]\n"
    "- cue: list of keywords (e.g., [\"because\", \"therefore\"]) or null\n"
    "- confidence: float 0.0-1.0\n\n"
    "EVENT 1: \"{cause_text}\"\n"
    "EVENT 2: \"{effect_text}\"\n\n"
    "Return ONLY the JSON object:"
)

PROMPT_SCENE_GROUPING = """You are a literary editor. You will be given a list of sequential events from a chapter.
Your job is to group these events into **narrative scenes**. A scene is a cluster of events that happen in the same general time/location and are thematically connected.

**Input:**
A JSON list of event objects:
[
    {{"id": "event/001", "name": "Pip enters graveyard"}},
    {{"id": "event/002", "name": "Magwitch confronts Pip"}},
    {{"id": "event/003", "name": "Magwitch threatens Pip"}},
    {{"id": "event/004", "name": "Pip runs home in fear"}},
    {{"id": "event/005", "name": "Pip enters the kitchen"}},
    {{"id": "event/006", "name": "Mrs. Joe scolds Pip"}}
]

**Output:**
Return a JSON object with a single "scenes" key. Each scene must have:
- event_ids: A list of event IDs that belong to this scene.
- theme: A short, 3-5 word theme for the scene (e.g., "A Fateful Encounter").
- confidence: Your 0.0-1.0 confidence in this grouping.

**Event List for Chapter {chapter_id}:**
{event_list_json}

Return ONLY the JSON object:"""

# --- END OF PROMPTS ---

def init_openai_client(api_key: str) -> openai.OpenAI:
    if openai is None:
        raise RuntimeError("openai package not installed. Run `pip install openai`.")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)

async def _async_llm_json_call(
    prompt: str, 
    model: str, 
    client: openai.OpenAI, 
    cache: BoundedCache, 
    cache_key: str,
    max_tokens: int = 4096,
    request_logprobs: bool = False
) -> Any:
    """
    A single, reusable, and robust function to call the LLM and get clean JSON.
    Handles caching, retries, JSON cleaning, and model-specific API params.
    """
    
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached, None 

    # Detect Model Type
    is_o1 = model.startswith("o1")
    is_gpt5 = model.startswith("gpt-5")

    for attempt in range(3):
        text = ""
        try:
            loop = asyncio.get_event_loop()
            
            def make_request():
                api_args = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                }

                if is_o1:
                    # o1: Strict rules. No system, no temperature, no logprobs.
                    # Must use max_completion_tokens. NO response_format.
                    api_args["max_completion_tokens"] = max_tokens
                elif is_gpt5:
                    # gpt-5: Likely supports JSON mode, uses max_completion_tokens.
                    api_args["max_completion_tokens"] = max_tokens
                    api_args["response_format"] = {"type": "json_object"}
                else:
                    # Standard (gpt-4o/3.5): max_tokens, temp, logprobs supported.
                    api_args["max_tokens"] = max_tokens
                    api_args["temperature"] = 0.0
                    api_args["response_format"] = {"type": "json_object"}
                    if request_logprobs:
                        api_args["logprobs"] = True

                return client.chat.completions.create(**api_args)
            
            resp = await loop.run_in_executor(None, make_request)
            text = resp.choices[0].message.content.strip()
            
            if not text:
                raise ExtractionError("LLM returned empty response")

            logprobs = getattr(resp.choices[0], "logprobs", None)
            
            # --- 1. Attempt Markdown Extraction ---
            # Matches ```json ... ``` OR just ``` ... ```
            json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()
            
            # --- 2. Attempt Raw JSON Extraction ---
            start = text.find("{")
            end = text.rfind("}") + 1
            
            if start != -1 and end > start:
                text = text[start:end]
            elif start != -1:
                # RECOVERY: Found a start '{' but no valid end '}' -> Truncated?
                # Let's try to salvage it by closing the JSON structure.
                # This is a "Hail Mary" pass for when tokens run out.
                print(f"[warning] JSON appears truncated. Attempting repair...")
                text = text[start:] + "]}" 
            else:
                # No JSON structure found at all
                print(f"[debug] No JSON found. Raw output start: {repr(text[:200])}")
                raise ExtractionError("No JSON brace found in response")

            # --- 3. Parse JSON ---
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Final attempt: sometimes repair needs an extra brace
                if text.endswith("]}"): 
                     pass
                # Try cleaning control characters as last resort
                cleaned_chars = [c for c in text if ord(c) >= 32 or c in '\n\r\t']
                text = ''.join(cleaned_chars)
                data = json.loads(text)
            
            # Unwrap common wrappers
            if isinstance(data, dict) and 'events' in data and isinstance(data['events'], list):
                data = data['events']
            elif isinstance(data, dict) and 'scenes' in data and isinstance(data['scenes'], list):
                data = data['scenes']

            await cache.set(cache_key, data)
            return data, logprobs
            
        except Exception as e:
            print(f"[error] LLM/JSON call failed (attempt {attempt+1}): {e}")
            if attempt == 2:
                raise ExtractionError(f"Extraction failed after 3 attempts: {e}")
            await asyncio.sleep(1 + attempt * 2)
    
    raise ExtractionError("Failed to get LLM response after retries")


def _get_extraction_prompt(
    text_input: str,
    chapter_id: int,
    extraction_style: str,
    enable_llm_expansion: bool
) -> str:
    if extraction_style == "high-level":
        instructions = PROMPT_HIGH_LEVEL_INSTRUCTIONS
    else: 
        instructions = PROMPT_DETAILED_INSTRUCTIONS
    
    if enable_llm_expansion:
        expansion_instructions = PROMPT_EXPANSION_ADDON
    else:
        expansion_instructions = ""
        
    return PROMPT_BASE_TEMPLATE.format(
        instructions=instructions,
        expansion_instructions=expansion_instructions,
        chapter_id=chapter_id, 
        text_input=text_input
    )

async def extract_events_from_text(
    text_input: str, 
    chapter_id: int, 
    model: str,
    client: openai.OpenAI,
    enable_llm_expansion: bool,
    request_logprobs: bool,
    extraction_style: str
) -> (Any, Any):
    
    cache_key = _hash_for_cache(
        f"{chapter_id}:{text_input}:{extraction_style}:{enable_llm_expansion}", 
        model
    )
    
    prompt = _get_extraction_prompt(
        text_input=text_input.strip(),
        chapter_id=chapter_id,
        extraction_style=extraction_style,
        enable_llm_expansion=enable_llm_expansion
    )

    # INCREASED TOKEN LIMIT TO 16k
    # 5 paragraphs of detailed extraction generates huge JSON.
    data, logprobs = await _async_llm_json_call(
        prompt=prompt,
        model=model,
        client=client,
        cache=event_extraction_cache,
        cache_key=cache_key,
        max_tokens=16000, 
        request_logprobs=request_logprobs
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
    extraction_style: str
) -> List[tuple[Any, Any]]:
    
    tasks = [
        extract_events_from_text(
            para, chapter_id, model, client, 
            enable_llm_expansion, request_logprobs, extraction_style
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


async def assess_causal_pair(
    cause_text: str, 
    effect_text: str, 
    model: str,
    client: openai.OpenAI
) -> Optional[Dict[str, Any]]:
    
    cache_key = _hash_for_cache(f"{cause_text}|||{effect_text}", model)
    prompt = PROMPT_CAUSAL_PAIR.format(cause_text=cause_text, effect_text=effect_text)

    try:
        data, _ = await _async_llm_json_call(
            prompt=prompt, model=model, client=client,
            cache=assessment_cache, cache_key=cache_key,
            max_tokens=1024 # Increased for safety
        )
        return data
    except Exception as e:
        print(f"[error] Causal assessment failed: {e}")
        return None

async def batch_assess_pairs(
    pairs: List[tuple[str, str, str, str]], 
    model: str,
    client: openai.OpenAI
) -> List[Optional[Dict[str, Any]]]:
    
    tasks = [
        assess_causal_pair(cause_text, effect_text, model, client)
        for cause_text, effect_text, _, _ in pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(None)
        else:
            processed_results.append(result)
    return processed_results

async def assess_semantic_pair_async(
    cause_text: str, 
    effect_text: str, 
    model: str,
    client: openai.OpenAI
) -> Optional[Dict[str, Any]]:
    
    cache_key = _hash_for_cache(f"sem:{cause_text}|||{effect_text}", model)
    prompt = PROMPT_SEMANTIC_PAIR.format(cause_text=cause_text, effect_text=effect_text)

    try:
        data, _ = await _async_llm_json_call(
            prompt=prompt, model=model, client=client,
            cache=semantic_cache, cache_key=cache_key,
            max_tokens=1024
        )
        return data
    except Exception as e:
        print(f"[error] Semantic assessment failed: {e}")
        return None

async def batch_assess_semantic_pairs(
    pairs: List[tuple[str, str, str, str]], 
    model: str,
    client: openai.OpenAI
) -> List[Optional[Dict[str, Any]]]:
    
    tasks = [
        assess_semantic_pair_async(cause_text, effect_text, model, client)
        for cause_text, effect_text, _, _ in pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(None)
        else:
            processed_results.append(result)
    return processed_results

async def extract_scenes_from_chapter_async(
    chapter_events: List[CEKEvent],
    chapter_id: int,
    model: str,
    client: openai.OpenAI
) -> List[Dict[str, Any]]:
    
    # Limit events to prevent context overflow
    if len(chapter_events) > 300:
         chapter_events = chapter_events[:300]

    event_list_simple = [
        {"id": ev.id, "name": ev.name, "seq": ev.sequence} 
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
            max_tokens=8192 # High limit for scenes
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