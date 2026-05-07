from neo4j import GraphDatabase
import os
import re

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "greeplace1!"  
DB_NAME = "20260326"        
FILE_PATH = "ge_import.txt"

def clean_statement(stmt):
    """Clean up Cypher statement"""
    if not stmt.strip():
        return None
    
    # Remove trailing commas before closing braces
    stmt = re.sub(r',\s*}', '}', stmt)
    stmt = stmt.strip()
    
    if not stmt.endswith(';'):
        stmt += ';'
    
    return stmt

def import_data():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: File not found: {FILE_PATH}")
        return

    print(f"🔌 Connecting to Neo4j...")
    
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ Connected")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print(f"📖 Reading {FILE_PATH}...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        script = f.read()

    # Split on semicolon + newline (preserves semicolons in text)
    queries = re.split(r';\s*[\r\n]+', script)
    queries = [clean_statement(q) for q in queries if q.strip()]
    
    # Filter executable statements
    executable = [q for q in queries 
                  if q and any(q.upper().strip().startswith(cmd) 
                  for cmd in ['CREATE', 'MERGE', 'MATCH', 'UNWIND'])]
    
    print(f"🚀 Executing {len(executable):,} statements...")
    
    success = 0
    failed = 0
    
    with driver.session(database=DB_NAME) as session:
        for i, query in enumerate(executable):
            try:
                session.run(query)
                success += 1
                
                if i % 100 == 0:
                    pct = 100 * (i + 1) / len(executable)
                    print(f"   {i+1}/{len(executable)} ({pct:.1f}%) | ✓{success} ✗{failed}", end="\r")
                    
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\n❌ Error: {str(e)[:100]}")

    print(f"\n\n✅ Complete: {success:,} succeeded, {failed:,} failed")
    driver.close()

if __name__ == "__main__":
    import_data()
