# run_cekg.py
import argparse
import sys
from pathlib import Path

# import the pipeline class from your file
from great_expectations_cekg_preprocessor import CEKGPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Run CEKG preprocessing on Great Expectations text")
    parser.add_argument("--input", "-i", required=True, help="Path to plaintext book (UTF-8)")
    parser.add_argument("--out-json", default="ge_preprocessed.json", help="JSON-LD output filename")
    parser.add_argument("--out-cypher", default="ge_import.cypher", help="Cypher script output filename")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model to load (e.g. en_core_web_sm or en_core_web_trf)")
    args = parser.parse_args()

    # lazy load spaCy model (the pipeline will accept nlp or fallback)
    try:
        import spacy
        nlp = spacy.load(args.model)
    except Exception as e:
        print("Warning: failed to load spaCy model:", e)
        print("Continuing with pipeline that may run with reduced capabilities.")
        nlp = None

    prep = CEKGPreprocessor(nlp=nlp)

    input_path = Path(args.input)
    if not input_path.exists():
        print("Input file not found:", input_path)
        sys.exit(2)

    out = prep.run(
        text_path=str(input_path),
        out_json=args.out_json,
        out_cypher=args.out_cypher
    )
    print("Done. Outputs:", out)

if __name__ == "__main__":
    main()
