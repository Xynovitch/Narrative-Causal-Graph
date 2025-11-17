import json
import asyncio
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
semantic_cache = BoundedCache(max_size=CACHE_MAX_SIZE) # New cache
scene_cache = BoundedCache(max_size=CACHE_MAX_SIZE) # New cache

# --- PROMPT SET 1: STANDARD (DEFAULT) ---

# --- PROMPT SET 1: STANDARD (DEFAULT) ---

# FIX: Removed 'f' prefix, changed {{chapter_id}} to {chapter_id}, kept JSON examples as {{...}}
PROMPT_PARAGRAPH_EXTRACTION = """You are an expert literary analyst extracting structured events from narrative text.
Extract EVERY distinct event from this paragraph. An event is ANY action, perception, dialogue, emotion, or state change.
Return a JSON object containing a single key "events". Each event in the array must be an object with:
- name: brief description
- eventType: one of ["perception/recognition", "encounter/dialogue", "conflict/threat", "assistance/exchange", "movement/departure", "reflection/emotion", "action", "other"]
- actionType: canonical verb
- actors: array of {{"name": "Pip", "strength": 0.0-1.0}} (use proper names, NEVER pronouns)
- patients: array of {{"name": "Magwitch", "strength": 0.0-1.0}}
- location: The specific place mentioned in the quote (e.g., "churchyard", "the forge"). If not explicitly mentioned, return null.
- time: temporal reference (if any)
- whyFactors: **This is mandatory.** Array of AT LEAST ONE {{"factor": "description", "weight": 0.0-1.0, "category": "Immediate Motivation/Instrumental Cause/Contextual Cause/Symbolic Cause"}}. If no motivation is explicit, infer the most likely "Contextual Cause" (e.g., "fear", "reacting to previous event", "survival instinct", "narrative progression").
- confidence: 0.0-1.0 (your confidence in the extraction)
- quote: the relevant sentence(s) from the text

PARAGRAPH (Chapter {chapter_id}):
{text_input}

JSON object:"""

PROMPT_CHUNK_EXTRACTION = """You are an expert literary analyst and narrative summarizer.
Read the following text chunk, which contains multiple paragraphs. Your goal is to extract only the **major, high-level narrative events or ideas**.
Do NOT extract every single verb or minor action. Focus on the main plot points, significant changes in state, or the core "idea-to-idea" flow of the narrative.
Return a JSON object containing a single key "events". Each event in the array must be an object with:
- name: brief description of the high-level event or idea.
- eventType: one of ["perception/recognition", "encounter/dialogue", "conflict/threat", "assistance/exchange", "movement/departure", "reflection/emotion", "action", "other"]
- actionType: canonical verb for the main idea (e.g., "realize", "decide", "confront").
- actors: array of {{"name": "Pip", "strength": 0.0-1.0}} (main actors involved).
- patients: array of {{"name": "Magwitch", "strength": 0.0-1.0}} (main entities affected).
- location: The primary location for this major event (if any).
- time: temporal reference (if any).
- whyFactors: **This is mandatory.** Array of AT LEAST ONE {{"factor": "description", "weight": 0.0-1.0, "category": "Immediate Motivation/Instrumental Cause/Contextual Cause/Symbolic Cause"}}.
- confidence: 0.0-1.0 (your confidence in the extraction)
- quote: A key representative sentence or phrase from the chunk (max 50 words).

TEXT CHUNK (Chapter {chapter_id}):
{text_input}

JSON object:"""

# --- PROMPT SET 2: EXPANDED (EXPERIMENTAL) ---

PROMPT_PARAGRAPH_EXPANDED = """You are an expert literary analyst extracting structured events from narrative text.
**Your Task:**
1.  **Extract All Events:** Find every action, perception, dialogue, emotion, or state change.
2.  **Find Implicit Events:** Extract psychological or emotional events (e.g., 'Pip felt ashamed').
3.  **Group Compound Events:** Combine simple, sequential actions into one event (e.g., 'walked to the door and knocked' -> "knock on door").
4.  **Coreference Resolution:** Normalize all actor/patient names to their **full, canonical name** (e.g., "the boy", "Philip Pirrip", "Pip" -> "Pip").

**Output Format:**
Return a JSON object containing a single key "events". Each event in the array must be an object with:
- name: brief description
- eventType: one of ["perception/recognition", "communication", "conflict", "assistance", "movement", "emotion", "transformation", "other"]
- actionType: canonical verb
- actors: array of {{"name": "Pip", "strength": 0.0-1.0}} (MUST be canonical full name)
- patients: array of {{"name": "Magwitch", "strength": 0.0-1.0}} (MUST be canonical full name)
- location: The specific place mentioned (e.g., "churchyard", "the forge").
- time: temporal reference (if any).
- whyFactors: **Mandatory.** Array of AT LEAST ONE {{"factor": "description", "weight": 0.0-1.0, "category": "Immediate/Contextual/Symbolic"}}.
- confidence: 0.0-1.0 (your confidence in the extraction)
- quote: the relevant sentence(s) from the text

PARAGRAPH (Chapter {chapter_id}):
{text_input}

JSON object:"""

PROMPT_CHUNK_EXPANDED = """You are an expert literary analyst and narrative summarizer.
**Your Task:**
1.  **Extract High-Level Events:** Read the text chunk and extract only the major narrative events or ideas.
2.  **Find Implicit Events:** Include major psychological or emotional turning points.
3.  **Coreference Resolution:** Normalize all actor/patient names to their **full, canonical name** (e.g., "the boy" -> "Pip").

**Output Format:**
Return a JSON object containing a single key "events". Each event in the array must be an object with:
- name: brief description of the high-level event or idea.
- eventType: one of ["perception/recognition", "communication", "conflict", "assistance", "movement", "emotion", "transformation", "other"]
- actionType: canonical verb for the main idea (e.g., "realize", "decide", "confront").
- actors: array of {{"name": "Pip", "strength": 0.0-1.0}} (MUST be canonical full name).
- patients: array of {{"name": "Magwitch", "strength": 0.0-1.0}} (MUST be canonical full name).
- location: The primary location for this major event (if any).
- time: temporal reference (if any).
- whyFactors: **Mandatory.** Array of AT LEAST ONE {{"factor": "description", "weight": 0.0-1.0, "category": "Immediate/Contextual/Symbolic"}}.
- confidence: 0.0-1.0 (your confidence in the extraction)
- quote: A key representative sentence or phrase from the chunk (max 50 words).

TEXT CHUNK (Chapter {chapter_id}):
{text_input}

JSON object:"""

# --- PROMPT SET 3: LINKING (CAUSAL & SEMANTIC) ---
# These were already correct (no 'f' prefix), but included for completeness
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

# --- PROMPT SET 4: SCENE GROUPING ---

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

**Example Output:**
{{
    "scenes": [
        {{
            "event_ids": ["event/001", "event/002", "event/003", "event/004"],
            "theme": "The Graveyard Threat",
            "confidence": 0.95
        }},
        {{
            "event_ids": ["event/005", "event/006"],
            "theme": "A Harsh Homecoming",
            "confidence": 0.90
        }}
    ]
}}

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
    Handles caching, retries, and JSON cleaning.
    """
    
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
                    temperature=0.0,
                    max_tokens=max_tokens,
                    logprobs=request_logprobs,
                )
            
            resp = await loop.run_in_executor(None, make_request)
            text = resp.choices[0].message.content.strip()
            logprobs = resp.choices[0].logprobs
            
            # 1. Strip markdown fences
            if text.startswith("```"):
                lines = text.split('\n')
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()
            
            # 2. Find JSON block
            if "{" not in text:
                raise ExtractionError(f"No JSON found in response (attempt {attempt+1})")
            
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

            # 3. Try to parse directly first (most responses won't need cleaning)
            try:
                data = json.loads(text)
                
                # Unwrap common response formats
                if isinstance(data, dict) and 'events' in data and isinstance(data['events'], list):
                    data = data['events']
                elif isinstance(data, dict) and 'scenes' in data and isinstance(data['scenes'], list):
                    data = data['scenes']
                
                await cache.set(cache_key, data)
                return data, logprobs
                
            except json.JSONDecodeError:
                # If direct parsing fails, try cleaning
                pass

            # 4. Clean control characters only if needed
            cleaned = []
            i = 0
            in_string = False
            
            while i < len(text):
                char = text[i]
                
                # Handle escape sequences
                if char == '\\' and i + 1 < len(text) and in_string:
                    next_char = text[i + 1]
                    # Keep valid JSON escape sequences
                    if next_char in '"\\/_bfnrtu':
                        cleaned.append(char)
                        cleaned.append(next_char)
                        i += 2
                        continue
                
                # Track string boundaries
                if char == '"' and (i == 0 or text[i-1] != '\\'):
                    in_string = not in_string
                    cleaned.append(char)
                    i += 1
                    continue
                
                # Inside strings: escape or skip control chars
                if in_string:
                    if char == '\n':
                        cleaned.append('\\n')
                    elif char == '\r':
                        cleaned.append('\\r')
                    elif char == '\t':
                        cleaned.append('\\t')
                    elif ord(char) < 32:
                        pass  # Skip other control chars
                    else:
                        cleaned.append(char)
                else:
                    # Outside strings: allow normal whitespace
                    if ord(char) >= 32 or char in '\n\r\t ':
                        cleaned.append(char)
                
                i += 1
            
            text = ''.join(cleaned)
            
            # 5. Parse the cleaned JSON
            data = json.loads(text)
            
            # Unwrap common response formats
            if isinstance(data, dict) and 'events' in data and isinstance(data['events'], list):
                data = data['events']
            elif isinstance(data, dict) and 'scenes' in data and isinstance(data['scenes'], list):
                data = data['scenes']
            
            await cache.set(cache_key, data)
            return data, logprobs
            
        except json.JSONDecodeError as e:
            error_context = text[:200] if 'text' in locals() else "N/A"
            print(f"[error] JSON decode failed (attempt {attempt+1}/{3}): {str(e)}")
            print(f"[debug] First 200 chars: {error_context}")
            if attempt == 2:
                raise ExtractionError(f"JSON parsing failed after 3 attempts. Last error: {str(e)}")
            await asyncio.sleep(1 + attempt * 2)
            
        except Exception as e:
            print(f"[error] LLM/JSON call failed (attempt {attempt+1}/{3}): {type(e).__name__}: {str(e)}")
            if attempt == 2:
                raise ExtractionError(f"Extraction failed after 3 attempts: {type(e).__name__}: {str(e)}")
            await asyncio.sleep(1 + attempt * 2)
    
    raise ExtractionError("Failed to get LLM response after retries")


async def extract_events_from_paragraph(
    paragraph: str, 
    chapter_id: int, 
    model: str,
    client: openai.OpenAI,
    enable_llm_expansion: bool = False,
    request_logprobs: bool = False
) -> (Any, Any):
    
    cache_key = _hash_for_cache(f"{chapter_id}:{paragraph}:{enable_llm_expansion}", model)
    
    if enable_llm_expansion:
        prompt_template = PROMPT_PARAGRAPH_EXPANDED
    else:
        prompt_template = PROMPT_PARAGRAPH_EXTRACTION
        
    prompt = prompt_template.format(
        chapter_id=chapter_id, 
        text_input=paragraph.strip()
    )

    data, logprobs = await _async_llm_json_call(
        prompt=prompt,
        model=model,
        client=client,
        cache=event_extraction_cache,
        cache_key=cache_key,
        max_tokens=3072, 
        request_logprobs=request_logprobs
    )
    
    if not isinstance(data, list):
        data = [data]
    
    return data, logprobs

async def extract_events_from_chunk_async(
    text_chunk: str, 
    chapter_id: int, 
    model: str,
    client: openai.OpenAI,
    enable_llm_expansion: bool = False,
    request_logprobs: bool = False
) -> (Any, Any):
    
    cache_key = _hash_for_cache(f"{chapter_id}:{text_chunk}:{enable_llm_expansion}", model)

    if enable_llm_expansion:
        prompt_template = PROMPT_CHUNK_EXPANDED
    else:
        prompt_template = PROMPT_CHUNK_EXTRACTION

    prompt = prompt_template.format(
        chapter_id=chapter_id,
        text_input=text_chunk.strip()
    )

    data, logprobs = await _async_llm_json_call(
        prompt=prompt,
        model=model,
        client=client,
        cache=event_extraction_cache,
        cache_key=cache_key,
        max_tokens=4096,
        request_logprobs=request_logprobs
    )
    
    if not isinstance(data, list):
        data = [data]
    
    return data, logprobs

async def batch_extract_events(
    paragraphs: List[tuple[str, int]], 
    model: str,
    client: openai.OpenAI,
    enable_llm_expansion: bool = False,
    request_logprobs: bool = False
) -> List[tuple[Any, Any]]:
    
    tasks = [
        extract_events_from_paragraph(
            para, chapter_id, model, client, 
            enable_llm_expansion, request_logprobs
        )
        for para, chapter_id in paragraphs
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Show the full error message
            error_msg = f"{type(result).__name__}: {str(result)}"
            print(f"[batch_extract] Paragraph {i} failed: {error_msg}")
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
            max_tokens=512
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

# --- NEW FUNCTIONS FOR SEMANTIC LINKING ---
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
            max_tokens=512
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
# --- END NEW FUNCTIONS ---

# --- NEW FUNCTION FOR SCENE GROUPING ---
async def extract_scenes_from_chapter_async(
    chapter_events: List[CEKEvent],
    chapter_id: int,
    model: str,
    client: openai.OpenAI
) -> List[Dict[str, Any]]:
    
    # Create a simplified JSON list of events for the prompt
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
            max_tokens=4096
        )
        # Data is already extracted from the 'scenes' wrapper by _async_llm_json_call
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[error] Scene grouping for chapter {chapter_id} failed: {e}")
        return []
# --- END NEW FUNCTION ---

async def get_cache_sizes() -> dict:
    return {
        "event_extraction_cache_size": await event_extraction_cache.size(),
        "assessment_cache_size": await assessment_cache.size(),
        "semantic_cache_size": await semantic_cache.size(),
        "scene_cache_size": await scene_cache.size()
    }