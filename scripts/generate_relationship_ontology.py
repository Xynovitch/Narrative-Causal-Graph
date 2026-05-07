import os
import glob
import json
import asyncio
import argparse
from typing import List, Set
from tqdm.asyncio import tqdm
import openai

# Reuse config if available
try:
    from cekg_pipeline import config
except ImportError:
    class MockConfig:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    config = MockConfig()

# -----------------------------------------------------------------------------
# 1. Relationship Theory Seed (Optional)
# -----------------------------------------------------------------------------
RELATIONSHIP_SEED = [
    # Causal / Logical
    "DIRECT_CAUSE", "ENABLES", "PREVENTS", "INHIBITS", "NECESSITATES",
    
    # Temporal / Structural
    "PRECEDES", "INTERRUPTS", "COINCIDES_WITH", "REPEATS",
    
    # Narrative / Thematic
    "FORESHADOWS", "MIRRORS", "CONTRASTS", "FULFILLS", "BETRAYS",
    
    # Character / Psychological
    "MOTIVATES", "PROVOKES", "INSPIRES", "CONFUSES", "ENRAGES"
]

# -----------------------------------------------------------------------------
# 2. Prompts
# -----------------------------------------------------------------------------

PROMPT_EXTRACT_RELATIONS = """
You are a narrative theorist. Read the following text.
Identify specific instances where one event influences another (Causal, Temporal, Thematic, Emotional).
Abstract these instances into **Relationship Categories**.

Examples:
- "Pip was scared, so he ran." -> EMOTIONAL_TRIGGER
- "The door was locked, preventing entry." -> PHYSICAL_BLOCKAGE
- "He planned the heist, then executed it." -> PLAN_EXECUTION

Rules:
1. Extract 5-10 distinct relationship types found in the text.
2. Use SNAKE_CASE (e.g., DIRECT_CAUSE, SOCIAL_PRESSURE).
3. Focus on the *nature of the link*, not the events themselves.

Text:
{text_chunk}

Return ONLY a valid JSON list of strings:
["TYPE_1", "TYPE_2"]
"""

PROMPT_CONSOLIDATE_RELATIONS = """
You are an expert ontology engineer.
I have collected a list of narrative relationship types.
Many are duplicates or too specific.

**Your Goal:**
Create a standardized "Relationship Type Dictionary" of exactly {target_size} items.

**Instructions:**
1. Deduplicate and merge synonyms (e.g., "SCARED_BY" and "FEAR_REACTION" -> "MOTIVATES" or "EMOTIONAL_TRIGGER").
2. Ensure you have categories for: Causal, Enabling, Preventing, Emotional, and Thematic links.
3. Output MUST be a clean JSON list of strings in SNAKE_CASE.

**Input Raw Types:**
{raw_list_str}

Return ONLY the JSON list:
"""

# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------

async def get_client():
    if not config.OPENAI_API_KEY:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return openai.AsyncOpenAI(api_key=api_key)
    return openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

async def call_llm(client, prompt: str) -> List[str]:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o", # Use a smart model for abstraction
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        raw_list = []
        
        # 1. Handle basic list vs dict structure
        if isinstance(data, list): 
            raw_list = data
        elif isinstance(data, dict):
            # Try to find a list value inside the dict
            for v in data.values():
                if isinstance(v, list): 
                    raw_list = v
                    break
            # Fallback: use keys if no list found
            if not raw_list:
                raw_list = list(data.keys())
        
        # 2. Sanitize items (Fix for AttributeError: 'dict' object has no attribute 'strip')
        clean_list = []
        for item in raw_list:
            if isinstance(item, str):
                clean_list.append(item)
            elif isinstance(item, dict):
                # If LLM returns [{"type": "CAUSE"}, ...], extract the first string value
                found_str = False
                for v in item.values():
                    if isinstance(v, str):
                        clean_list.append(v)
                        found_str = True
                        break
                if not found_str:
                    # Fallback: stringify the dict keys or the dict itself (last resort)
                    clean_list.append(str(item))
            else:
                clean_list.append(str(item))
                
        return clean_list

    except Exception as e:
        print(f"[Error] LLM Call failed: {e}")
        return []

async def process_file(client, file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except: return []

    # Sample 3 chunks
    chunk_size = 3000
    if len(text) < chunk_size: chunks = [text]
    else:
        indices = [int(i * (len(text) - chunk_size) / 3) for i in range(3)]
        chunks = [text[i:i+chunk_size] for i in indices]

    tasks = [call_llm(client, PROMPT_EXTRACT_RELATIONS.format(text_chunk=c)) for c in chunks]
    results = await asyncio.gather(*tasks)
    
    # Flatten results
    return [item for sublist in results for item in sublist]

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="relationship_ontology.json")
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--include-theory", action="store_true")
    args = parser.parse_args()

    client = await get_client()
    files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    
    if not files:
        print(f"No .txt files found in {args.input_dir}")
        return

    print(f"Found {len(files)} novels. Starting extraction...")
    
    # 1. Extraction
    sem = asyncio.Semaphore(5)
    async def safe_process(f):
        async with sem: return await process_file(client, f)
    
    tasks = [safe_process(f) for f in files]
    results = await tqdm.gather(*tasks)
    
    # 2. Aggregation
    raw_pool = set()
    if args.include_theory:
        raw_pool.update(RELATIONSHIP_SEED)
        
    for res in results:
        for t in res: 
            # Safe because 'call_llm' guarantees strings now
            if isinstance(t, str):
                raw_pool.add(t.strip().upper().replace(" ", "_"))
        
    print(f"Collected {len(raw_pool)} raw relationship types. Consolidating...")

    # 3. Consolidation
    raw_list = list(raw_pool)
    prompt = PROMPT_CONSOLIDATE_RELATIONS.format(
        target_size=args.target_size,
        raw_list_str=json.dumps(raw_list)
    )
    final_ontology = await call_llm(client, prompt)
    
    # 4. Save
    final_unique = sorted(list(set(final_ontology)))
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({"relationship_types": final_unique}, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(final_unique)} types to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())