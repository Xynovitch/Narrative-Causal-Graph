import re
import os
from neo4j import GraphDatabase

# --- CONFIGURATION ---
# UPDATE THESE TO MATCH YOUR ACTUAL SETTINGS
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "greeplace1!"      # <--- YOUR PASSWORD HERE
DB_NAME = "test2"      # <--- YOUR DATABASE NAME HERE (e.g. "neo4j", "test", "graph.db")
FILE_PATH = "ge_import.txt"

def clean_and_import():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: Could not find {FILE_PATH}")
        return

    print(f"📖 Reading {FILE_PATH}...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # --- STEP 1: FIX SYNTAX ERRORS ---
    print("✨ Scrubbing syntax errors (trailing commas)...")
    
    # Regex to remove trailing commas in maps: {key: 'val', } -> {key: 'val'}
    fixed_content = re.sub(r",\s*\}", "}", content)
    
    # Regex to remove trailing commas in lists: [val, ] -> [val]
    fixed_content = re.sub(r",\s*\]", "]", fixed_content)

    # --- STEP 2: SPLIT INTO STATEMENTS ---
    # Split by semicolon ONLY if it's at the end of a line (preserves text semicolons)
    statements = re.split(r';\s*[\r\n]+', fixed_content)
    statements = [s.strip() for s in statements if s.strip()]
    
    print(f"🚀 Found {len(statements)} statements to execute.")

    # --- STEP 3: CONNECT AND RUN ---
    print(f"🔌 Connecting to Neo4j at {URI} (Database: {DB_NAME})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    try:
        driver.verify_connectivity()
        # Connect specifically to the defined database
        with driver.session(database=DB_NAME) as session:
            for i, query in enumerate(statements):
                try:
                    # Skip empty or comment-only blocks to be safe
                    if not query.upper().startswith(("UNWIND", "CREATE", "MERGE", "MATCH", "CALL", "LOAD")):
                        continue

                    session.run(query)
                    
                    if i % 10 == 0:
                        print(f"   Processed {i}/{len(statements)}...", end="\r")
                except Exception as e:
                    print(f"\n❌ Error on statement {i}: {e}")
                    print(f"   Query snippet: {query[:100]}...")

        print(f"\n✅ Import Complete! {len(statements)} statements executed.")
        
        # --- STEP 4: SAFETY STITCHING ---
        print("🧵 Running safety stitch (linking sequential events)...")
        with driver.session(database=DB_NAME) as session:
            stitch_query = """
            MATCH (s:Scene)-[:INCLUDES]->(e:Event)
            WITH s, e ORDER BY e.sequence ASC
            WITH s, collect(e) as events
            UNWIND range(0, size(events)-2) as i
            WITH events[i] as current, events[i+1] as next
            MERGE (current)-[:PRECEDES]->(next)
            """
            session.run(stitch_query)
            print("✅ Events successfully linked chronologically.")

    except Exception as e:
        print(f"\n❌ Critical Connection Error: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    clean_and_import()