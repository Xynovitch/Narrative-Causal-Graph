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
             "Set to 0 to process the ENTIRE chapter in one call. "
             "Default is 1 (one call per paragraph)."
    )
    parser.add_argument(
        "--extraction-style",
        choices=["detailed", "high-level"],
        default="detailed",
        help="[EXPERIMENTAL] 'detailed' = Verb-to-Verb (Default). 'high-level' = Idea-to-Idea."
    )
    parser.add_argument(
        "--graph-model",
        choices=["chain", "star"],
        default="chain",
        help="[EXPERIMENTAL] 'chain' = Event->Entity->Event (Default). 'star' = Entity->[Events]."
    )

    # --- NEW EXPERIMENTAL ARGUMENTS (SCHEMA) ---
    parser.add_argument("--enable-scene-grouping", action="store_true")
    parser.add_argument("--enable-semantic-linking", action="store_true")
    parser.add_argument("--enable-llm-expansion", action="store_true")
    parser.add_argument("--enable-confidence-calibration", action="store_true")

    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='sk-...'")
        exit(1)

    try:
        preprocessor = CEKGPreprocessor(openai_model=args.openai_model)
        
        print(f"[config] model: {preprocessor.openai_model}")
        print(f"[config] batch_size: {args.batch_size}")
        print(f"[config] style: {args.extraction_style} (Verb-to-Verb if detailed)")
        
        if args.paragraph_chunk_size != 1:
            chunk_str = "ALL" if args.paragraph_chunk_size == 0 else str(args.paragraph_chunk_size)
            print(f"[config] EXPERIMENTAL: Processing {chunk_str} paragraphs per call.")

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
            # --- Pass new args ---
            paragraph_chunk_size=args.paragraph_chunk_size,
            extraction_style=args.extraction_style,
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
        # ... (standard summary printing) ...
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ FATAL ERROR: {e}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        exit(1)

if __name__ == "__main__":
    main()