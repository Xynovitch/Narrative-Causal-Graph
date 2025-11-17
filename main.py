import os
import time
import argparse
import traceback
import asyncio
from cekg_pipeline.pipeline import CEKGPreprocessor
from cekg_pipeline.config import (
    OPENAI_API_KEY, OPENAI_MODEL, BATCH_SIZE, 
    CAUSAL_BATCH_SIZE, SAMPLE_RATE, CACHE_MAX_SIZE
)

# Find the absolute path of the project's root directory (where main.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="CEKG Preprocessor (DUAL FLOW)")
    
    # --- Input/Output Arguments ---
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--out-json", default=os.path.join(PROJECT_ROOT, "ge_preprocessed.json"))
    parser.add_argument("--out-cypher", default=os.path.join(PROJECT_ROOT, "ge_import.cypher"))
    parser.add_argument("--out-csv", default=os.path.join(PROJECT_ROOT, "neo4j_csv"))
    
    # --- Model & Batching Arguments ---
    parser.add_argument("--openai-model", default=OPENAI_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of paragraphs to process in parallel (default: 5)")
    parser.add_argument("--causal-batch-size", type=int, default=CAUSAL_BATCH_SIZE)
    parser.add_argument("--causal-window", type=int, default=4)
    parser.add_argument("--causal-sample-rate", type=float, default=SAMPLE_RATE)
    parser.add_argument("--max-chapters", type=int, default=None)

    # --- EXPERIMENTAL ARGUMENTS (GROUP 0) ---
    parser.add_argument(
        "--paragraph-chunk-size", 
        type=int, 
        default=1,
        help="[EXPERIMENTAL] Group N paragraphs into a single text chunk for one API call. "
             "Default is 1 (one call per paragraph, processed in parallel batches)."
    )
    parser.add_argument(
        "--graph-model",
        choices=["chain", "star"],
        default="chain",
        help="[EXPERIMENTAL] Set graph output model. "
             "'chain' (default): Event->Entity->Event. "
             "'star': Canonical Entity->[Events]"
    )

    # --- NEW EXPERIMENTAL ARGUMENTS (SCHEMA) ---
    parser.add_argument(
        "--enable-scene-grouping",
        action="store_true",
        help="[EXPERIMENTAL] Enable Group 1: Scene Grouping. Runs an extra LLM pass to group events into scenes."
    )
    parser.add_argument(
        "--enable-semantic-linking",
        action="store_true",
        help="[EXPERIMENTAL] Enable Group 1: Semantic Cohesion. Runs an extra LLM pass to find non-causal links."
    )
    parser.add_argument(
        "--enable-llm-expansion",
        action="store_true",
        help="[EXPERIMENTAL] Enable Group 2: LLM Expansion. Modifies prompts for coreference and implicit event extraction."
    )
    parser.add_argument(
        "--enable-confidence-calibration",
        action="store_true",
        help="[EXPERIMENTAL] Enable Group 3: Calibrated Confidence. (Requires 'sentence-transformers')."
    )
    # --- END NEW ARGUMENTS ---

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
        
        # --- Print experimental flags if used ---
        if args.paragraph_chunk_size > 1:
            print(f"[config] EXPERIMENTAL: Paragraph chunk size set to {args.paragraph_chunk_size}.")
        if args.graph_model == "star":
            print(f"[config] EXPERIMENTAL: Graph model set to 'star'.")
        if args.enable_scene_grouping:
            print(f"[config] EXPERIMENTAL: Scene Grouping enabled.")
        if args.enable_semantic_linking:
            print(f"[config] EXPERIMENTAL: Semantic Linking enabled.")
        if args.enable_llm_expansion:
            print(f"[config] EXPERIMENTAL: LLM Expansion (Coreference/Implicit Events) enabled.")
        if args.enable_confidence_calibration:
            print(f"[config] EXPERIMENTAL: Calibrated Confidence enabled.")
        # ---

        start_time = time.time()
        
        input_path = args.input
        if not os.path.isabs(input_path):
            input_path = os.path.join(PROJECT_ROOT, input_path)

        out = asyncio.run(preprocessor.run_async(
            text_path=input_path, 
            out_json=args.out_json, 
            out_cypher=args.out_cypher, 
            out_csv_dir=args.out_csv,
            max_chapters=args.max_chapters,
            batch_size=args.batch_size,
            causal_window=args.causal_window,
            causal_sample_rate=args.causal_sample_rate,
            causal_batch_size=args.causal_batch_size,
            # --- Pass new args to the pipeline ---
            paragraph_chunk_size=args.paragraph_chunk_size,
            graph_model=args.graph_model,
            enable_scene_grouping=args.enable_scene_grouping,
            enable_semantic_linking=args.enable_semantic_linking,
            enable_llm_expansion=args.enable_llm_expansion,
            enable_confidence_calibration=args.enable_confidence_calibration
        ))
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✓ Finished in {elapsed:.2f} seconds")
        print(f"✓ Events: {out['stats']['events']}")
        print(f"✓ Event→Entity productions: {out['stats']['event_produces_entity']}")
        print(f"✓ Entity→Event links: {out['stats']['entity_points_to_event']}")
        print(f"✓ Causal links (Event→Event): {out['stats']['causal_links']}")
        # --- Add new stats ---
        if 'semantic_links' in out['stats']:
            print(f"✓ Semantic links (Event→Event): {out['stats']['semantic_links']}")
        if 'scenes' in out['stats']:
            print(f"✓ Scenes created: {out['stats']['scenes']}")
        # ---
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