#!/usr/bin/env python3
"""
Fix event_category references in pipeline.py
The schema was updated but the pipeline still tries to use event_category
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create backup of file before modifying"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"  ✓ Backed up to {backup_path}")
        return backup_path
    return None

def fix_pipeline():
    """Fix pipeline.py to use action_type instead of event_category"""
    print("\n[FIX] Updating pipeline.py to use action_type...")
    
    file_path = "cekg_pipeline/pipeline.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The LLM returns "event_category" in JSON
    # But our schema only has "action_type"
    # Solution: Read event_category from JSON, use it as action_type
    
    # Find the line where event_type is extracted
    old_extraction = r'event_type = event_data\.get\("event_category", "OTHER"\)'
    new_extraction = 'event_type = event_data.get("event_category", event_data.get("action_type", "OTHER"))'
    
    content = re.sub(old_extraction, new_extraction, content)
    
    # Also update the CEKEvent initialization
    # Find the CEKEvent creation block
    old_event_init = r'event = schemas\.CEKEvent\(\s*id=utils\._make_id\("event"\),\s*raw_description=event_data\.get\("raw_description".*?\),\s*event_category=event_type,'
    
    new_event_init = '''event = schemas.CEKEvent(
                id=utils._make_id("event"),
                raw_description=event_data.get("raw_description", event_data.get("name", "Untitled")),
                action_type=event_type,'''
    
    content = re.sub(old_event_init, new_event_init, content, flags=re.DOTALL)
    
    # Also remove any other references to event_category in CEKEvent()
    content = re.sub(r'event_category=', 'action_type=', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  ✓ Updated pipeline.py to use action_type")
    print("  ✓ Now reads event_category from LLM but stores as action_type")

def fix_graph_mapper():
    """Ensure graph_mapper uses action_type, not event_category"""
    print("\n[FIX] Updating graph_mapper.py...")
    
    file_path = "cekg_pipeline/graph_mapper.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace any reference to ev.event_category with ev.action_type
    content = content.replace('ev.event_category', 'ev.action_type')
    
    # Ensure "category" is not in the properties
    # Remove the category line if it exists
    content = re.sub(r'"category": ev\.action_type,\s*', '', content)
    content = re.sub(r'"category": ev\.event_category,\s*', '', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  ✓ Updated graph_mapper.py")

def verify_schema():
    """Verify that schemas.py has action_type and not event_category"""
    print("\n[VERIFY] Checking schemas.py...")
    
    with open("cekg_pipeline/schemas.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find CEKEvent dataclass
    cekevent_start = content.find("@dataclass\nclass CEKEvent:")
    if cekevent_start == -1:
        print("  ✗ Could not find CEKEvent class")
        return False
    
    cekevent_section = content[cekevent_start:cekevent_start+1000]
    
    has_action_type = "action_type: str" in cekevent_section
    has_event_category = "event_category: str" in cekevent_section
    
    print(f"  action_type present: {'✓' if has_action_type else '✗'}")
    print(f"  event_category present: {'✗' if not has_event_category else '✓ (SHOULD BE REMOVED)'}")
    
    if has_event_category:
        print("\n  ! Warning: event_category still in schema")
        print("  ! Run the complete_fix.py script first")
        return False
    
    if not has_action_type:
        print("\n  ! Error: action_type missing from schema")
        return False
    
    print("  ✓ Schema is correct")
    return True

def main():
    print("="*60)
    print("Fix event_category References")
    print("="*60)
    
    try:
        # First verify schema is correct
        if not verify_schema():
            print("\n! Please run complete_fix.py first to fix schemas.py")
            return 1
        
        fix_pipeline()
        fix_graph_mapper()
        
        print("\n" + "="*60)
        print("✅ Fixed event_category references")
        print("="*60)
        print("\nChanges:")
        print("  1. pipeline.py now reads 'event_category' from LLM JSON")
        print("  2. But stores it as 'action_type' in CEKEvent")
        print("  3. graph_mapper.py uses ev.action_type")
        print("  4. No more 'unexpected keyword argument' errors")
        print("\nYou can now run:")
        print("  python main.py --input great_expectations.txt --full")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())