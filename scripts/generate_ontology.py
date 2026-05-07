import os
import glob
import json
import asyncio
import argparse
from typing import List, Set
from tqdm.asyncio import tqdm
import openai

# Try to reuse existing pipeline config, otherwise mock it
try:
    from cekg_pipeline import config, llm_service
except ImportError:
    class MockConfig:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_MODEL = "gpt-4o"
    config = MockConfig()

# -----------------------------------------------------------------------------
# 1. Narrative Theory Seed Data (Optional)
# -----------------------------------------------------------------------------
NARRATIVE_THEORY_SEED = [
    # John Truby (The Anatomy of Story)
    "WEAKNESS_REVELATION", "GHOST_CONFRONTATION", "INCITING_INCIDENT", 
    "DESIRE_FORMATION", "ALLY_GATHERING", "PLAN_EXECUTION", 
    "OPPONENT_CONFRONTATION", "DRIVE_FOR_GOAL", "ATTACK_BY_ALLY", 
    "APPARENT_DEFEAT", "VISIT_TO_DEATH", "BATTLE", "SELF_REVELATION", 
    "MORAL_DECISION", "NEW_EQUILIBRIUM",
    
    # Robert McKee (Story)
    "INCITING_EVENT", "PROGRESSIVE_COMPLICATION", "CRISIS_POINT", 
    "CLIMACTIC_ACTION", "RESOLUTION_EVENT", "VALUE_CHARGE_SHIFT",
    "TURNING_POINT", "SETUP_PAYOFF", "REVERSAL_OF_FORTUNE",
    
    # General Structural
    "INTRODUCTION", "EXPOSITION", "RISING_ACTION", "FALLING_ACTION",
    "DEUS_EX_MACHINA", "CLIFFHANGER", "FLASHBACK", "FORESHADOWING"
]

# -----------------------------------------------------------------------------
# 2. Prompts
# -----------------------------------------------------------------------------

PROMPT_EXTRACT_RAW_TYPES = """
You are a literary taxonomist. Read the following excerpt from a novel.
Identify abstract, distinct "Narrative Event Types" present in the text.
Do not extract specific events (e.g., "Pip eats pie"). Instead, extract the CATEGORY (e.g., "INGESTION_OF_FOOD", "THEFT", "FEAR_REACTION").

**Rules:**
1. Use SNAKE_CASE for categories (e.g., PHYSICAL_CONFLICT).
2. Be specific but reusable across different stories.
3. Extract 10-20 types.

**Text Excerpt:**
{text_chunk}

Return ONLY a valid JSON list of strings:
["CATEGORY_ONE", "CATEGORY_TWO", ...]
"""

PROMPT_CONSOLIDATE_TYPES = """
You are an expert ontology engineer for narrative AI.
I have collected a list of raw event types from various novels.
Many are duplicates, synonyms, or too specific.

**Your Goal:**
Create a standardized "Event Type Dictionary" of exactly {target_size} items.

**Instructions:**
1. Deduplicate and merge synonyms (e.g., "WALKING", "RUNNING" -> "PHYSICAL_MOVEMENT").
2. Ensure the list covers: Action, Dialogue, Thought, Emotion, Social Interaction, and Plot Structure.
3. Output MUST be a clean JSON list of strings in SNAKE_CASE.

**Input Raw Types:**
{raw_list_str}

Return ONLY the JSON list:
"""

# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------

async def get_openai_client():
    if not config.OPENAI_API_KEY:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in config or environment variables.")
        return openai.AsyncOpenAI(api_key=api_key)
    return openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)

async def call_llm(client, prompt: str, model: str = "gpt-4o-mini") -> List[str]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        if isinstance(data, list): return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list): return v
            return list(data.keys())
            
        return []
    except Exception as e:
        print(f"[Error] LLM Call failed: {e}")
        return []

async def process_novel_file(client, file_path: str) -> List[str]:
    """Reads a novel file and extracts types from chunks."""
    print(f"Processing: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return []

    # Sample text chunks
    chunk_size = 3000
    total_len = len(text)
    
    if total_len < chunk_size:
        chunks = [text]
    else:
        indices = [int(i * (total_len - chunk_size) / 5) for i in range(5)]
        chunks = [text[i : i + chunk_size] for i in indices]

    tasks = [
        call_llm(client, PROMPT_EXTRACT_RAW_TYPES.format(text_chunk=chunk))
        for chunk in chunks
    ]
    
    results = await asyncio.gather(*tasks)
    
    extracted_types = []
    for res in results:
        extracted_types.extend(res)
        
    return extracted_types

async def main():
    parser = argparse.ArgumentParser(description="Generate Narrative Event Ontology")
    parser.add_argument("--input-dir", required=True, help="Folder containing .txt novel files")
    parser.add_argument("--output", default="event_ontology.json", help="Output JSON file path")
    parser.add_argument("--target-size", type=int, default=100, help="Number of types in final dictionary")
    # NEW ARGUMENT: Optional inclusion of theory terms
    parser.add_argument("--include-theory", action="store_true", help="Include narrative theory terms in the seed")
    args = parser.parse_args()

    client = await get_openai_client()
    
    files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    if not files:
        print(f"No .txt files found in {args.input_dir}.")
        return

    print(f"Found {len(files)} novels. Starting extraction...")

    sem = asyncio.Semaphore(5) 
    
    async def safe_process(f):
        async with sem:
            return await process_novel_file(client, f)

    tasks = [safe_process(f) for f in files]
    results = await tqdm.gather(*tasks)

    # 3. Aggregate Raw Types
    raw_pool: Set[str] = set()
    
    # Conditional inclusion of seed data
    if args.include_theory:
        print("Including Narrative Theory Seed terms...")
        raw_pool.update(NARRATIVE_THEORY_SEED)
    else:
        print("Skipping Narrative Theory Seed terms (PURE DATA-DRIVEN MODE)...")

    for res in results:
        for t in res:
            raw_pool.add(t.strip().upper().replace(" ", "_"))
            
    print(f"Collected {len(raw_pool)} unique raw types. Consolidating...")

    # 4. Consolidate into clean Dictionary
    raw_list = list(raw_pool)
    final_ontology = []
    
    batch_size = 300 
    consolidated_batches = []
    
    print(f"Consolidating in {len(raw_list)//batch_size + 1} batches...")
    
    for i in range(0, len(raw_list), batch_size):
        batch = raw_list[i : i + batch_size]
        sub_target = max(20, args.target_size // (len(raw_list)//batch_size + 1))
        
        prompt = PROMPT_CONSOLIDATE_TYPES.format(
            target_size=sub_target, 
            raw_list_str=json.dumps(batch)
        )
        res = await call_llm(client, prompt, model="gpt-4o")
        consolidated_batches.extend(res)

    print("Finalizing list...")
    if len(consolidated_batches) > args.target_size:
        prompt = PROMPT_CONSOLIDATE_TYPES.format(
            target_size=args.target_size,
            raw_list_str=json.dumps(consolidated_batches)
        )
        final_ontology = await call_llm(client, prompt, model="gpt-4o")
    else:
        final_ontology = consolidated_batches

    # 5. Save
    final_unique = sorted(list(set(final_ontology)))
    
    output_data = {
        "metadata": {
            "source_novels_count": len(files),
            "requested_size": args.target_size,
            "actual_size": len(final_unique),
            "theory_included": args.include_theory
        },
        "event_types": final_unique
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(final_unique)} event types to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())