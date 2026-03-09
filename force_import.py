from neo4j import GraphDatabase
import os
import re  # <--- Added Regex module

# CONFIGURATION
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "greeplace1!"  
DB_NAME = "test"        
FILE_PATH = "ge_import.txt"

def import_data():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: Could not find {FILE_PATH}")
        return

    print(f"🔌 Connecting to Neo4j at {URI}...")
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ Connected successfully.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print(f"📖 Reading {FILE_PATH}...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        cypher_script = f.read()

    # --- THE FIX IS HERE ---
    # Only split on semicolons that are followed by a newline.
    # This preserves semicolons inside your story text.
    queries = re.split(r';\s*[\r\n]+', cypher_script)
    queries = [q.strip() for q in queries if q.strip()]
    # -----------------------

    print(f"🚀 Found {len(queries)} Cypher statements to execute.")

    with driver.session(database=DB_NAME) as session:
        for i, query in enumerate(queries):
            try:
                # Basic check to skip empty or comment-only blocks
                if not query.upper().startswith(("UNWIND", "CREATE", "MERGE", "MATCH", "CALL", "LOAD")):
                    continue
                    
                session.run(query)
                if i % 10 == 0:
                    print(f"   Processed {i}/{len(queries)} statements...", end="\r")
            except Exception as e:
                print(f"\n❌ Error on statement {i}: {e}")
                # We print a short snippet to identify the bad query
                print(f"Query snippet: {query[:100]}...\n")

    print(f"\n✅ Import Complete! {len(queries)} statements scanned.")
    driver.close()

if __name__ == "__main__":
    import_data()