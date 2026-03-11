#!/usr/bin/env python3
"""
Check what agent types are defined in your schema
"""

import json
import os

def check_schema():
    """Check schema.json for agent types"""
    
    # Common schema file locations
    possible_paths = [
        "schema.json",
        "schemas/schema.json",
        "cekg_pipeline/schema.json",
        "config/schema.json"
    ]
    
    schema_path = None
    for path in possible_paths:
        if os.path.exists(path):
            schema_path = path
            break
    
    if not schema_path:
        print("❌ No schema.json file found!")
        print("\nSearched in:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nIf you don't have a schema file, the pipeline uses defaults.")
        print("Default agent types include:")
        print("  - PROTAGONIST_HERO")
        print("  - MORAL_ANTAGONIST")
        print("  - ALLY_MENTOR")
        print("  - STRUCTURAL_AGENT")
        print("  - CRISIS_FORCER")
        return
    
    print(f"✓ Found schema: {schema_path}\n")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        schema = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print("\nTrying to fix concatenated JSON objects...")
        
        # Try to fix multiple root objects
        import re
        formatted = re.sub(r'\}\s*\{', '},{', content)
        formatted = f"[{formatted}]"
        
        try:
            list_of_dicts = json.loads(formatted)
            schema = {}
            for d in list_of_dicts:
                schema.update(d)
            print("✓ Fixed and merged JSON\n")
        except:
            print("❌ Could not fix JSON")
            return
    
    # Look for AgentTypeDictionary
    agent_types = schema.get("AgentTypeDictionary", [])
    
    if not agent_types:
        print("⚠️  No AgentTypeDictionary found in schema")
        print("   Pipeline will use default agent types\n")
        agent_types = [
            {"name": "PROTAGONIST_HERO", "theory": "@McKee"},
            {"name": "MORAL_ANTAGONIST", "theory": "@McKee"},
            {"name": "ALLY_MENTOR", "theory": "@McKee"},
            {"name": "STRUCTURAL_AGENT", "theory": "@McKee"},
            {"name": "CRISIS_FORCER", "theory": "@McKee"}
        ]
        print("Default Agent Types:")
    else:
        print(f"✓ Found {len(agent_types)} agent types in schema:\n")
        print("Agent Types Defined:")
    
    for agent in agent_types:
        name = agent.get("name", "???")
        theory = agent.get("theory", "???")
        desc = agent.get("description", agent.get("explanation", ""))
        print(f"  • {name} ({theory})")
        if desc:
            print(f"    {desc[:80]}...")
    
    # Check if PROTAGONIST_HERO is present
    print("\n" + "="*60)
    agent_names = [a.get("name") for a in agent_types]
    
    if "PROTAGONIST_HERO" in agent_names:
        print("✅ PROTAGONIST_HERO is defined")
        print("   Pip can be classified as PROTAGONIST_HERO")
    else:
        print("❌ PROTAGONIST_HERO is NOT defined")
        print("   Pip will NOT be classified as PROTAGONIST_HERO")
        print("\n   Add this to your schema.json AgentTypeDictionary:")
        print('''
{
  "name": "PROTAGONIST_HERO",
  "theory": "@McKee",
  "description": "The main character who drives the story and undergoes transformation"
}
        ''')
    
    print("="*60)

def add_protagonist_to_schema():
    """Helper to add PROTAGONIST_HERO if missing"""
    
    possible_paths = ["schema.json", "schemas/schema.json"]
    schema_path = None
    for path in possible_paths:
        if os.path.exists(path):
            schema_path = path
            break
    
    if not schema_path:
        print("\n❌ No schema.json found to update")
        return
    
    print(f"\n[FIX] Adding PROTAGONIST_HERO to {schema_path}...")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        schema = json.loads(content)
    except:
        # Try fixing concatenated JSON
        import re
        formatted = re.sub(r'\}\s*\{', '},{', content)
        formatted = f"[{formatted}]"
        list_of_dicts = json.loads(formatted)
        schema = {}
        for d in list_of_dicts:
            schema.update(d)
    
    # Add PROTAGONIST_HERO if not present
    agent_types = schema.get("AgentTypeDictionary", [])
    agent_names = [a.get("name") for a in agent_types]
    
    if "PROTAGONIST_HERO" not in agent_names:
        new_agent = {
            "name": "PROTAGONIST_HERO",
            "theory": "@McKee",
            "description": "The main character who drives the story and undergoes transformation",
            "explanation": "Central character through whose eyes we experience the narrative"
        }
        agent_types.append(new_agent)
        schema["AgentTypeDictionary"] = agent_types
        
        # Save back
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        print("✓ Added PROTAGONIST_HERO to schema")
    else:
        print("✓ PROTAGONIST_HERO already in schema")

def main():
    print("="*60)
    print("Schema Agent Type Checker")
    print("="*60)
    print()
    
    check_schema()
    
    print("\nOptions:")
    print("  1. If PROTAGONIST_HERO is missing, run:")
    print("     python check_schema_agent_types.py --add-protagonist")
    print("  2. Or manually add it to your schema.json")
    print("  3. Or run without a schema (uses defaults with PROTAGONIST_HERO)")
    
    # Check for --add-protagonist flag
    import sys
    if "--add-protagonist" in sys.argv:
        add_protagonist_to_schema()

if __name__ == "__main__":
    main()