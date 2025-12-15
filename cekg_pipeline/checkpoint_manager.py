"""
Checkpoint Manager for CEKG Pipeline
Provides robust saving and recovery for long-running processing
"""
import os
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

class CheckpointManager:
    """
    Manages pipeline checkpoints for recovery and resumption.
    
    Checkpoint Structure:
    - Stage metadata (name, timestamp, status)
    - Data serialization (JSON for readability, Pickle for complex objects)
    - Validation hashes
    - Progress tracking
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", run_id: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or use provided run ID
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.checkpoint_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoint metadata
        self.metadata_file = self.run_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        print(f"[checkpoint] Initialized: {self.run_dir}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load or create metadata file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "stages": {},
            "last_checkpoint": None
        }
    
    def _save_metadata(self):
        """Save metadata file"""
        self.metadata["updated_at"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash for data validation"""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def save_checkpoint(
        self, 
        stage: str, 
        data: Dict[str, Any],
        description: str = ""
    ) -> bool:
        """
        Save a checkpoint for a specific stage.
        
        Args:
            stage: Stage name (e.g., "extraction", "causal_linking")
            data: Data to save (will be serialized)
            description: Human-readable description
        
        Returns:
            bool: True if save successful
        """
        try:
            checkpoint_file = self.run_dir / f"{stage}.checkpoint"
            
            # Prepare checkpoint data
            checkpoint = {
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "data": data
            }
            
            # Compute validation hash
            data_hash = self._compute_hash(data)
            checkpoint["hash"] = data_hash
            
            # Save with pickle (handles complex objects)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Also save JSON version (for inspection)
            json_file = self.run_dir / f"{stage}.json"
            try:
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                print(f"[warning] Could not save JSON version: {e}")
            
            # Update metadata
            self.metadata["stages"][stage] = {
                "saved_at": checkpoint["timestamp"],
                "description": description,
                "hash": data_hash,
                "file": str(checkpoint_file)
            }
            self.metadata["last_checkpoint"] = stage
            self._save_metadata()
            
            print(f"[checkpoint] ✓ Saved: {stage} ({data_hash})")
            return True
            
        except Exception as e:
            print(f"[checkpoint] ✗ Failed to save {stage}: {e}")
            return False
    
    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for a specific stage.
        
        Args:
            stage: Stage name to load
        
        Returns:
            Data dictionary or None if not found/invalid
        """
        try:
            checkpoint_file = self.run_dir / f"{stage}.checkpoint"
            
            if not checkpoint_file.exists():
                print(f"[checkpoint] No checkpoint found for: {stage}")
                return None
            
            # Load checkpoint
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Validate hash
            expected_hash = checkpoint.get("hash")
            actual_hash = self._compute_hash(checkpoint["data"])
            
            if expected_hash != actual_hash:
                print(f"[checkpoint] ✗ Hash mismatch for {stage}")
                print(f"  Expected: {expected_hash}")
                print(f"  Actual: {actual_hash}")
                return None
            
            print(f"[checkpoint] ✓ Loaded: {stage} (from {checkpoint['timestamp']})")
            return checkpoint["data"]
            
        except Exception as e:
            print(f"[checkpoint] ✗ Failed to load {stage}: {e}")
            return None
    
    def has_checkpoint(self, stage: str) -> bool:
        """Check if a checkpoint exists for a stage"""
        checkpoint_file = self.run_dir / f"{stage}.checkpoint"
        return checkpoint_file.exists()
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints"""
        return list(self.metadata.get("stages", {}).keys())
    
    def get_last_checkpoint(self) -> Optional[str]:
        """Get the name of the last saved checkpoint"""
        return self.metadata.get("last_checkpoint")
    
    def clear_checkpoint(self, stage: str) -> bool:
        """Delete a specific checkpoint"""
        try:
            checkpoint_file = self.run_dir / f"{stage}.checkpoint"
            json_file = self.run_dir / f"{stage}.json"
            
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            if json_file.exists():
                json_file.unlink()
            
            if stage in self.metadata["stages"]:
                del self.metadata["stages"][stage]
            
            if self.metadata.get("last_checkpoint") == stage:
                remaining = self.list_checkpoints()
                self.metadata["last_checkpoint"] = remaining[-1] if remaining else None
            
            self._save_metadata()
            print(f"[checkpoint] Cleared: {stage}")
            return True
            
        except Exception as e:
            print(f"[checkpoint] Failed to clear {stage}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Delete all checkpoints for this run"""
        try:
            import shutil
            shutil.rmtree(self.run_dir)
            print(f"[checkpoint] Cleared all checkpoints for run: {self.run_id}")
            return True
        except Exception as e:
            print(f"[checkpoint] Failed to clear run: {e}")
            return False
    
    def get_progress_summary(self) -> str:
        """Get a human-readable summary of progress"""
        stages = self.metadata.get("stages", {})
        
        if not stages:
            return "No checkpoints yet"
        
        lines = [f"Run ID: {self.run_id}"]
        lines.append(f"Created: {self.metadata.get('created_at', 'unknown')}")
        lines.append(f"\nCompleted Stages ({len(stages)}):")
        
        for stage, info in stages.items():
            lines.append(f"  ✓ {stage}")
            lines.append(f"    - Saved: {info['saved_at']}")
            if info.get('description'):
                lines.append(f"    - {info['description']}")
        
        last = self.metadata.get("last_checkpoint")
        if last:
            lines.append(f"\nLast Checkpoint: {last}")
        
        return "\n".join(lines)


def serialize_events(events: List) -> List[Dict]:
    """Serialize event objects for checkpointing"""
    return [asdict(e) for e in events]


def deserialize_events(data: List[Dict], event_class) -> List:
    """Deserialize event objects from checkpoint"""
    return [event_class(**d) for d in data]


def serialize_links(links: List) -> List[Dict]:
    """Serialize link objects for checkpointing"""
    return [asdict(link) for link in links]


def deserialize_links(data: List[Dict], link_class) -> List:
    """Deserialize link objects from checkpoint"""
    return [link_class(**d) for d in data]


# ============================================================================
# Example Usage in Pipeline
# ============================================================================

"""
In pipeline.py, you would use it like this:

async def run_async(self, ...):
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir="./checkpoints",
        run_id=f"novel_{os.path.basename(text_path)}"
    )
    
    # Check for existing progress
    if checkpoint_mgr.has_checkpoint("extraction"):
        print("[resume] Found existing extraction checkpoint")
        data = checkpoint_mgr.load_checkpoint("extraction")
        all_events = deserialize_events(data["events"], schemas.CEKEvent)
        all_produces = deserialize_links(data["produces"], schemas.EventProducesEntity)
        entity_occurrences = data["entity_occurrences"]
    else:
        # Do extraction
        all_events, all_produces, entity_occurrences = await extract_events(...)
        
        # Save checkpoint
        checkpoint_mgr.save_checkpoint(
            "extraction",
            {
                "events": serialize_events(all_events),
                "produces": serialize_links(all_produces),
                "entity_occurrences": dict(entity_occurrences)
            },
            description=f"Extracted {len(all_events)} events"
        )
    
    # Continue with next stage...
    if checkpoint_mgr.has_checkpoint("causal_linking"):
        data = checkpoint_mgr.load_checkpoint("causal_linking")
        causal_links = deserialize_links(data["causal_links"], schemas.CausalLink)
    else:
        causal_links = await link_events(...)
        checkpoint_mgr.save_checkpoint(
            "causal_linking",
            {"causal_links": serialize_links(causal_links)},
            description=f"Created {len(causal_links)} causal links"
        )
"""