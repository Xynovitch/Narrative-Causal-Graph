"""
Integrated Main Script - Smart Linking + Semantic Analysis
Simplified arguments, better defaults for cost optimization
"""
import os
import time
import argparse
import traceback
import asyncio
from cekg_pipeline.pipeline import CEKGPreprocessor
from cekg_pipeline.config import OPENAI_API_KEY, OPENAI_MODEL

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        description="CEKG Preprocessor - Integrated Smart Linking + Semantic Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode (no semantic, medium granularity)
  python main.py --input novel.txt --fast

  # Full mode (everything including semantic, medium granularity)
  python main.py --input novel.txt --full

  # High granularity (like old paragraph-by-paragraph, ~300 events/chapter)
  python main.py --input novel.txt --chunk-size 1500

  # Ultra-high granularity (maximum detail, higher cost)
  python main.py --input novel.txt --chunk-size 800

  # Low granularity (fast and cheap, ~50 events/chapter)
  python main.py --input novel.txt --chunk-size 6000

  # Custom: High granularity + semantic + mixed theory
  python main.py --input novel.txt --chunk-size 1500 --enable-semantic-linking --enable-mixed-theory

Chunk Size Guide:
  800-1200:  Ultra-high granularity (~400-500 events/chapter, higher cost)
  1500-2000: High granularity (~250-350 events/chapter, moderate cost)
  3000-4000: Medium granularity (~100-150 events/chapter, balanced) [DEFAULT]
  5000-8000: Low granularity (~30-60 events/chapter, lowest cost)
        """
    )
    
    # --- Input/Output ---
    parser.add_argument("--input", "-i", required=True, 
                       help="Input text file path")
    parser.add_argument("--out-json", default=os.path.join(PROJECT_ROOT, "ge_preprocessed.json"))
    parser.add_argument("--out-cypher", default=os.path.join(PROJECT_ROOT, "ge_import.cypher"))
    parser.add_argument("--out-csv", default=os.path.join(PROJECT_ROOT, "neo4j_csv"))
    
    # --- Schema ---
    parser.add_argument("--schema-path", default=None, 
                       help="Path to JSON schema file for ontologies")
    
    # --- Model ---
    parser.add_argument("--openai-model", default=OPENAI_MODEL,
                       help="OpenAI model to use (default: gpt-4o-mini)")
    
    # --- Processing Limits ---
    parser.add_argument("--max-chapters", type=int, default=None,
                       help="Limit number of chapters to process")
    parser.add_argument("--max-pairs", type=int, default=5000,
                       help="Maximum causal pairs to evaluate (default: 5000, recommended: 3000-8000)")
    parser.add_argument("--max-concurrent-calls", type=int, default=10,
                       help="Concurrent API calls (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=3000,
                       help="Characters per extraction chunk (default: 3000, lower=more granular, higher=cheaper)")
    
    # --- Graph Model ---
    parser.add_argument("--graph-model", choices=["chain", "star"], default="star",
                       help="Graph structure (default: star)")
    
    # --- Preset Modes ---
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument("--fast", action="store_true",
                            help="Fast mode: Minimal features, lowest cost (~$2 per novel)")
    preset_group.add_argument("--full", action="store_true",
                            help="Full mode: All features enabled including semantic (~$5 per novel)")
    
    # --- Individual Feature Flags ---
    parser.add_argument("--enable-scene-grouping", action="store_true",
                       help="Group events into narrative scenes")
    parser.add_argument("--enable-agent-classification", action="store_true",
                       help="Classify character agent types")
    parser.add_argument("--enable-confidence-calibration", action="store_true",
                       help="Use multi-signal confidence scoring")
    parser.add_argument("--enable-mixed-theory", action="store_true", default=True,
                       help="Use both McKee and Truby theories")
    parser.add_argument("--disable-mixed-theory", action="store_true",
                       help="Use only McKee theory")
    parser.add_argument("--enable-semantic-linking", action="store_true",
                       help="Extract semantic relationships (explanation, contrast, etc.)")
    parser.add_argument("--disable-semantic-linking", action="store_true",
                       help="Disable semantic relationship extraction")

    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("\n" + "="*60)
        print("ERROR: OPENAI_API_KEY not set!")
        print("="*60)
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=sk-...")
        print("="*60 + "\n")
        exit(1)

    # Apply presets
    if args.fast:
        print("\n[mode] FAST MODE - Minimal features, lowest cost")
        enable_scene_grouping = False
        enable_agent_classification = False
        enable_confidence_calibration = False
        enable_semantic_linking = False
        max_pairs = 3000
    elif args.full:
        print("\n[mode] FULL MODE - All features enabled")
        enable_scene_grouping = True
        enable_agent_classification = True
        enable_confidence_calibration = True
        enable_semantic_linking = True
        max_pairs = args.max_pairs
    else:
        # Use individual flags
        enable_scene_grouping = args.enable_scene_grouping
        enable_agent_classification = args.enable_agent_classification
        enable_confidence_calibration = args.enable_confidence_calibration
        enable_semantic_linking = args.enable_semantic_linking and not args.disable_semantic_linking
        max_pairs = args.max_pairs

    enable_mixed_theory = args.enable_mixed_theory and not args.disable_mixed_theory

    try:
        preprocessor = CEKGPreprocessor(
            openai_model=args.openai_model,
            schema_path=args.schema_path
        )
        
        print("\n" + "="*60)
        print("INTEGRATED CEKG PIPELINE")
        print("Smart Linking + Semantic Analysis")
        print("="*60)
        print(f"Model: {preprocessor.openai_model}")
        print(f"Graph: {args.graph_model}")
        print(f"Theory: {'Mixed (McKee + Truby)' if enable_mixed_theory else 'McKee only'}")
        print(f"Chunk Size: {args.chunk_size} chars (~{args.chunk_size//800} paragraphs)")
        print(f"Max Pairs: {max_pairs:,}")
        print(f"Scene Grouping: {'✓' if enable_scene_grouping else '✗'}")
        print(f"Agent Classification: {'✓' if enable_agent_classification else '✗'}")
        print(f"Confidence Calibration: {'✓' if enable_confidence_calibration else '✗'}")
        print(f"Semantic Linking: {'✓' if enable_semantic_linking else '✗'}")
        print("="*60 + "\n")

        start_time = time.time()
        
        input_path = args.input
        if not os.path.isabs(input_path):
            input_path = os.path.join(PROJECT_ROOT, input_path)

        if not os.path.exists(input_path):
            print(f"\n[ERROR] Input file not found: {input_path}\n")
            exit(1)

        out = asyncio.run(preprocessor.run_async(
            text_path=input_path, 
            out_json=args.out_json, 
            out_cypher=args.out_cypher, 
            out_csv_dir=args.out_csv,
            max_chapters=args.max_chapters,
            graph_model=args.graph_model,
            enable_scene_grouping=enable_scene_grouping,
            enable_agent_classification=enable_agent_classification,
            enable_confidence_calibration=enable_confidence_calibration,
            enable_mixed_theory=enable_mixed_theory,
            enable_semantic_linking=enable_semantic_linking,
            max_concurrent_calls=args.max_concurrent_calls,
            max_long_range_pairs=max_pairs,
            chunk_size=args.chunk_size
        ))
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("✓ PROCESSING COMPLETE")
        print("="*60)
        print(f"Time: {elapsed:.2f}s")
        print(f"Events: {out['stats']['events']:,}")
        print(f"Characters: {out['stats']['characters']:,}")
        print(f"Causal Links: {out['stats']['causal_links']:,}")
        
        if enable_mixed_theory:
            print(f"  - McKee: {out['stats'].get('mckee_links', 0):,}")
            print(f"  - Truby: {out['stats'].get('truby_links', 0):,}")
        
        if enable_semantic_linking and 'semantic_links' in out['stats']:
            print(f"Semantic Links: {out['stats']['semantic_links']:,}")
        
        if enable_scene_grouping and 'scenes' in out['stats']:
            print(f"Scenes: {out['stats']['scenes']:,}")
        
        if enable_agent_classification and 'agent_types_classified' in out['stats']:
            print(f"Agent Types: {out['stats']['agent_types_classified']:,}")
        
        print(f"\nDAG Valid: {'✓' if out['stats']['dag_valid'] else '✗'}")
        
        # Cost estimate
        events = out['stats']['events']
        links = out['stats']['causal_links']
        chapters = args.max_chapters if args.max_chapters else 20
        
        # Rough cost calculation
        extraction_cost = chapters * 0.05  # ~$0.05 per chapter
        
        # Adjust extraction cost based on chunk size
        # Smaller chunks = more API calls but better granularity
        chunk_multiplier = 3000 / args.chunk_size  # baseline is 3000
        extraction_cost *= chunk_multiplier
        
        causal_cost = (max_pairs / 50) * 0.001  # Bulk batches
        scene_cost = chapters * 0.01 if enable_scene_grouping else 0
        agent_cost = out['stats'].get('agent_types_classified', 0) * 0.005 if enable_agent_classification else 0
        semantic_cost = 0  # FREE (piggybacks on causal calls + local embeddings)
        
        total_cost = extraction_cost + causal_cost + scene_cost + agent_cost
        
        print(f"\n[cost estimate]")
        print(f"  Extraction: ~${extraction_cost:.2f}")
        print(f"  Causal: ~${causal_cost:.2f}")
        if enable_scene_grouping:
            print(f"  Scenes: ~${scene_cost:.2f}")
        if enable_agent_classification:
            print(f"  Agents: ~${agent_cost:.2f}")
        if enable_semantic_linking:
            print(f"  Semantic: $0.00 (FREE - piggybacks on causal + local embeddings)")
        print(f"  TOTAL: ~${total_cost:.2f}")
        
        if enable_semantic_linking:
            print(f"\n  ✓ Semantic analysis adds ZERO cost!")
        
        print("="*60 + "\n")
        
        print(f"Output files:")
        print(f"  JSON: {args.out_json}")
        print(f"  Cypher: {args.out_cypher}")
        print(f"  CSV: {args.out_csv}/")
        
        if enable_semantic_linking:
            print(f"\n✓ Semantic relationships exported to CSVs")
        print()
        
    except KeyboardInterrupt:
        print("\n\n[interrupted] Processing cancelled by user\n")
        exit(1)
    except Exception as e:
        print("\n" + "="*60)
        print("✗ FATAL ERROR")
        print("="*60)
        print(f"\n{e}\n")
        traceback.print_exc()
        print("\n" + "="*60 + "\n")
        exit(1)

if __name__ == "__main__":
    main()