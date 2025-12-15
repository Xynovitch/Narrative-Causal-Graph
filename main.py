"""
Main Script with Checkpoint Support
Adds --resume and --clear-checkpoints flags
"""
import os
import time
import argparse
import traceback
import asyncio
from cekg_pipeline.pipeline import CEKGPreprocessor
from cekg_pipeline.checkpoint_manager import CheckpointManager
from cekg_pipeline.config import OPENAI_API_KEY, OPENAI_MODEL

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        description="CEKG Preprocessor with Checkpoint Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh
  python main.py --input novel.txt --full

  # Resume from last checkpoint
  python main.py --input novel.txt --full --resume

  # Clear checkpoints and start fresh
  python main.py --input novel.txt --full --clear-checkpoints

  # Use custom checkpoint directory
  python main.py --input novel.txt --checkpoint-dir ./my_checkpoints --resume
        """
    )
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True, help="Input text file path")
    parser.add_argument("--out-json", default=os.path.join(PROJECT_ROOT, "ge_preprocessed.json"))
    parser.add_argument("--out-cypher", default=os.path.join(PROJECT_ROOT, "ge_import.cypher"))
    parser.add_argument("--out-csv", default=os.path.join(PROJECT_ROOT, "neo4j_csv"))
    
    # Checkpoint Management
    checkpoint_group = parser.add_argument_group('Checkpoint Options')
    checkpoint_group.add_argument("--checkpoint-dir", default="./checkpoints",
                                 help="Directory for checkpoint files (default: ./checkpoints)")
    checkpoint_group.add_argument("--resume", action="store_true",
                                 help="Resume from last checkpoint if available")
    checkpoint_group.add_argument("--no-checkpoints", action="store_true",
                                 help="Disable checkpoint system entirely")
    checkpoint_group.add_argument("--clear-checkpoints", action="store_true",
                                 help="Clear existing checkpoints before starting")
    checkpoint_group.add_argument("--list-checkpoints", action="store_true",
                                 help="List available checkpoints and exit")
    
    # Schema
    parser.add_argument("--schema-path", default=None, 
                       help="Path to JSON schema file for ontologies")
    
    # Model
    parser.add_argument("--openai-model", default=OPENAI_MODEL,
                       help="OpenAI model to use (default: gpt-4o-mini)")
    
    # Processing Limits
    parser.add_argument("--max-chapters", type=int, default=None,
                       help="Limit number of chapters to process")
    parser.add_argument("--max-pairs", type=int, default=5000,
                       help="Maximum causal pairs to evaluate (default: 5000)")
    parser.add_argument("--max-concurrent-calls", type=int, default=10,
                       help="Concurrent API calls (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=3000,
                       help="Characters per extraction chunk (default: 3000)")
    
    # Graph Model
    parser.add_argument("--graph-model", choices=["chain", "star"], default="star",
                       help="Graph structure (default: star)")
    
    # Preset Modes
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument("--fast", action="store_true",
                            help="Fast mode: Minimal features (~$2/novel)")
    preset_group.add_argument("--full", action="store_true",
                            help="Full mode: All features including semantic (~$5/novel)")
    
    # Individual Feature Flags
    parser.add_argument("--enable-scene-grouping", action="store_true")
    parser.add_argument("--enable-agent-classification", action="store_true")
    parser.add_argument("--enable-confidence-calibration", action="store_true")
    parser.add_argument("--enable-mixed-theory", action="store_true", default=True)
    parser.add_argument("--disable-mixed-theory", action="store_true")
    parser.add_argument("--enable-semantic-linking", action="store_true")
    parser.add_argument("--disable-semantic-linking", action="store_true")

    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("\n" + "="*60)
        print("ERROR: OPENAI_API_KEY not set!")
        print("="*60)
        print("\nPlease set your OpenAI API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("="*60 + "\n")
        exit(1)

    # Handle checkpoint listing
    if args.list_checkpoints:
        input_path = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            exit(1)
        
        run_id = f"{os.path.basename(input_path)}_{int(os.path.getmtime(input_path))}"
        checkpoint_mgr = CheckpointManager(checkpoint_dir=args.checkpoint_dir, run_id=run_id)
        
        print("\n" + "="*60)
        print("CHECKPOINT STATUS")
        print("="*60)
        print(checkpoint_mgr.get_progress_summary())
        print("="*60 + "\n")
        exit(0)

    # Apply presets
    if args.fast:
        enable_scene_grouping = False
        enable_agent_classification = False
        enable_confidence_calibration = False
        enable_semantic_linking = False
        max_pairs = 3000
    elif args.full:
        enable_scene_grouping = True
        enable_agent_classification = True
        enable_confidence_calibration = True
        enable_semantic_linking = True
        max_pairs = args.max_pairs
    else:
        enable_scene_grouping = args.enable_scene_grouping
        enable_agent_classification = args.enable_agent_classification
        enable_confidence_calibration = args.enable_confidence_calibration
        enable_semantic_linking = args.enable_semantic_linking and not args.disable_semantic_linking
        max_pairs = args.max_pairs

    enable_mixed_theory = args.enable_mixed_theory and not args.disable_mixed_theory
    enable_checkpoints = not args.no_checkpoints

    try:
        # Initialize preprocessor
        preprocessor = CEKGPreprocessor(
            openai_model=args.openai_model,
            schema_path=args.schema_path,
            checkpoint_dir=args.checkpoint_dir,
            enable_checkpoints=enable_checkpoints
        )
        
        # Resolve input path
        input_path = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
        if not os.path.exists(input_path):
            print(f"\n[ERROR] Input file not found: {input_path}\n")
            exit(1)
        
        # Handle checkpoint clearing
        if args.clear_checkpoints and enable_checkpoints:
            run_id = f"{os.path.basename(input_path)}_{int(os.path.getmtime(input_path))}"
            checkpoint_mgr = CheckpointManager(checkpoint_dir=args.checkpoint_dir, run_id=run_id)
            checkpoint_mgr.clear_all()
            print("[checkpoint] Cleared all existing checkpoints\n")
        
        # Show configuration
        print("\n" + "="*60)
        print("INTEGRATED CEKG PIPELINE")
        print("="*60)
        print(f"Model: {preprocessor.openai_model}")
        print(f"Graph: {args.graph_model}")
        print(f"Chunk Size: {args.chunk_size}")
        print(f"Max Pairs: {max_pairs:,}")
        print(f"Checkpoints: {'✓' if enable_checkpoints else '✗'}")
        if enable_checkpoints:
            print(f"  Directory: {args.checkpoint_dir}")
            print(f"  Resume: {'✓' if args.resume else '✗'}")
        print(f"\nFeatures:")
        print(f"  Theory: {'Mixed' if enable_mixed_theory else 'McKee only'}")
        print(f"  Scene Grouping: {'✓' if enable_scene_grouping else '✗'}")
        print(f"  Agent Classification: {'✓' if enable_agent_classification else '✗'}")
        print(f"  Confidence Calibration: {'✓' if enable_confidence_calibration else '✗'}")
        print(f"  Semantic Linking: {'✓' if enable_semantic_linking else '✗'}")
        print("="*60 + "\n")

        start_time = time.time()
        
        # Run pipeline
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
            chunk_size=args.chunk_size,
            resume_from_checkpoint=args.resume
        ))
        
        elapsed = time.time() - start_time
        
        # Show results
        print("\n" + "="*60)
        print("✓ PROCESSING COMPLETE")
        print("="*60)
        print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
        print(f"Events: {out['stats']['events']:,}")
        print(f"Characters: {out['stats']['characters']:,}")
        print(f"Causal Links: {out['stats']['causal_links']:,}")
        
        if enable_mixed_theory:
            print(f"  - McKee: {out['stats'].get('mckee_links', 0):,}")
            print(f"  - Truby: {out['stats'].get('truby_links', 0):,}")
        
        if enable_semantic_linking:
            print(f"Semantic Links: {out['stats']['semantic_links']:,}")
        
        if enable_scene_grouping:
            print(f"Scenes: {out['stats']['scenes']:,}")
        
        if enable_agent_classification:
            print(f"Agent Types: {out['stats']['agent_types_classified']:,}")
        
        print(f"\nDAG Valid: {'✓' if out['stats']['dag_valid'] else '✗'}")
        print("="*60 + "\n")
        
        print(f"Output files:")
        print(f"  JSON: {args.out_json}")
        print(f"  Cypher: {args.out_cypher}")
        print(f"  CSV: {args.out_csv}/")
        print()
        
    except KeyboardInterrupt:
        print("\n\n[interrupted] Processing cancelled by user")
        print("[info] Progress saved in checkpoints. Use --resume to continue.\n")
        exit(1)
    except Exception as e:
        print("\n" + "="*60)
        print("✗ FATAL ERROR")
        print("="*60)
        print(f"\n{e}\n")
        traceback.print_exc()
        print("\n" + "="*60)
        print("[info] Check checkpoints with: python main.py --input novel.txt --list-checkpoints")
        print("[info] Resume with: python main.py --input novel.txt --resume")
        print("="*60 + "\n")
        exit(1)

if __name__ == "__main__":
    main()