import json
import asyncio
from typing import List, Dict, Optional, Any

try:
    import openai
except ImportError:
    openai = None

from .schemas import ExtractionError
from .utils import BoundedCache, _hash_for_cache
from .config import CACHE_MAX_SIZE

# --- Caches are now local to this service ---
event_extraction_cache = BoundedCache(max_size=CACHE_MAX_SIZE)
assessment_cache = BoundedCache(max_size=CACHE_MAX_SIZE)

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
    max_tokens: int = 3072,
    json_start: str = "[",
    json_end: str = "]"
) -> Any:
    """
    A single, reusable, and robust function to call the LLM and get clean JSON.
    Handles caching, retries, and JSON cleaning.
    """
    
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached

    for attempt in range(3):
        try:
            loop = asyncio.get_event_loop()
            
            def make_request():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            
            resp = await loop.run_in_executor(None, make_request)
            text = resp.choices[0].message.content.strip()
            
            # 1. Strip markdown
            if text.startswith("```"):
                lines = text.split('\n')
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()
            
            text = text.replace('\r\n', '\n').replace('\r', '\n')

            # 2. Find JSON block
            if json_start not in text or json_end not in text:
                if "{" in text and "}" in text: # Fallback for single object
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    text = "[" + text[start:end] + "]" # Wrap in list to be safe
                else:
                    raise ExtractionError(f"No JSON found in response")
            else:
                start = text.find(json_start)
                end = text.rfind(json_end) + 1
                text = text[start:end]

            # 3. Clean control characters
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
                        pass # Skip unprintable control character
                    else:
                        cleaned_chars.append(char)
                else:
                    if ord(char) >= 32 or char in '\n\r\t ':
                        cleaned_chars.append(char)
                
                i += 1
            
            text = ''.join(cleaned_chars)
            
            data = json.loads(text)
            await cache.set(cache_key, data)
            return data
            
        except Exception as e:
            print(f"[error] LLM/JSON call failed (attempt {attempt+1}): {e}")
            if attempt == 2:
                raise ExtractionError(f"Extraction failed after 3 attempts: {e}")
            await asyncio.sleep(1 + attempt * 2)
    
    raise ExtractionError("Failed to get LLM response after retries")


async def extract_events_from_paragraph(
    paragraph: str, 
    chapter_id: int, 
    model: str,
    client: openai.OpenAI
) -> List[Dict[str, Any]]:
    
    cache_key = _hash_for_cache(f"{chapter_id}:{paragraph}", model)
    
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

    result = await _async_llm_json_call(
        prompt=prompt,
        model=model,
        client=client,
        cache=event_extraction_cache,
        cache_key=cache_key,
        max_tokens=3072,
        json_start="[",
        json_end="]"
    )
    
    if not isinstance(result, list):
        result = [result]
    
    return result

async def batch_extract_events(
    paragraphs: List[tuple[str, int]], 
    model: str,
    client: openai.OpenAI
) -> List[List[Dict[str, Any]]]:
    
    tasks = [
        extract_events_from_paragraph(para, chapter_id, model, client)
        for para, chapter_id in paragraphs
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"[batch_extract] Paragraph {i} failed: {result}")
            processed_results.append([]) # Return empty list on failure
        else:
            processed_results.append(result)
    
    return processed_results


async def assess_causal_pair(
    cause_quote: str, 
    effect_quote: str, 
    model: str,
    client: openai.OpenAI
) -> Optional[Dict[str, Any]]:
    
    cache_key = _hash_for_cache(f"{cause_quote}|||{effect_quote}", model)
    
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

    try:
        result = await _async_llm_json_call(
            prompt=prompt,
            model=model,
            client=client,
            cache=assessment_cache,
            cache_key=cache_key,
            max_tokens=512,
            json_start="{",
            json_end="}"
        )
        return result
    except Exception as e:
        print(f"[error] Causal assessment failed: {e}")
        return None # Return None on failure

async def batch_assess_pairs(
    pairs: List[tuple[str, str, str, str]], 
    model: str,
    client: openai.OpenAI
) -> List[Optional[Dict[str, Any]]]:
    
    tasks = [
        assess_causal_pair(cause_quote, effect_quote, model, client)
        for cause_quote, effect_quote, _, _ in pairs
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append(None)
        else:
            processed_results.append(result)
            
    return processed_results

async def get_cache_sizes() -> dict:
    return {
        "event_extraction_cache_size": await event_extraction_cache.size(),
        "assessment_cache_size": await assessment_cache.size()
    }