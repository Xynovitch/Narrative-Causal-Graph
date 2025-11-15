import os
import time
import argparse
import traceback
import asyncio
from cekg_pipeline.pipeline import CEKGPreprocessor
# --- FIX IS HERE ---
# Changed 'CAUSAL_SAMPLE_RATE' to 'SAMPLE_RATE' to match config.py
from cekg_pipeline.config import (
    OPENAI_API_KEY, OPENAI_MODEL, BATCH_SIZE, 
    CAUSAL_BATCH_SIZE, SAMPLE_RATE, CACHE_MAX_SIZE
)
# --- END FIX ---

def main():
    parser = argparse.ArgumentParser(description="CEKG Preprocessor (DUAL FLOW)")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--out-json", default="ge_preprocessed.json")
    parser.add_argument("--out-cypher", default="ge_import.cypher")
    parser.add_argument("--out-csv", default="neo4j_csv")
    parser.add_argument("--openai-model", default=OPENAI_MODEL)
    parser.add_argument("--max-chapters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--causal-batch-size", type=int, default=CAUSAL_BATCH_SIZE)
    parser.add_argument("--causal-window", type=int, default=4)
    
    # --- FIX IS HERE ---
    # Set the default from 'SAMPLE_RATE'
    parser.add_argument("--causal-sample-rate", type=float, default=SAMPLE_RATE)
    # --- END FIX ---

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
        
        out = asyncio.run(preprocessor.run_async(
            text_path=args.input, 
            out_json=args.out_json, 
            out_cypher=args.out_cypher, 
            out_csv_dir=args.out_csv,
            max_chapters=args.max_chapters,
            batch_size=args.batch_size,
            causal_window=args.causal_window,
            causal_sample_rate=args.causal_sample_rate,
            causal_batch_size=args.causal_batch_size
        ))
        
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

if __name__ == "__main__":
    main()