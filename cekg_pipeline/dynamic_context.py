"""
Dynamic context windows for causal linking.

Uses the local thematic engine (embeddings) as a calculation machine to find
long-shot event pairs (e.g. event 99 and 3023) with high thematic similarity,
instead of fixed windows (e.g. first 500 / last 500). A double sliding window
moves in opposite directions; only pairs above a thematic similarity threshold
(default 0.95) are kept, minimizing events sent to the remote LLM.

- Long-range: double sliding window + 0.95 thematic similarity.
- Local: chronologically adjacent events and events within a small window
  (or same scene when scenes are provided) are always considered for linking.
"""

from typing import List, Tuple, Set, Dict, Optional, Any
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBED_AVAILABLE = True
except ImportError:
    _EMBED_AVAILABLE = False


# Defaults: high bar for long-shot pairs to minimize API calls
THEMATIC_SIMILARITY_THRESHOLD = 0.95
LOCAL_WINDOW_SIZE = 5  # events within this distance always checked
DOUBLE_WINDOW_SIZE = 100  # events per sliding window
DOUBLE_WINDOW_STEP = 50


def _get_embedding_model() -> Optional[Any]:
    if not _EMBED_AVAILABLE:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def get_long_shot_pairs_double_sliding(
    events: List[Any],
    event_map: Dict[str, Any],
    similarity_threshold: float = THEMATIC_SIMILARITY_THRESHOLD,
    window_size: int = DOUBLE_WINDOW_SIZE,
    step: int = DOUBLE_WINDOW_STEP,
    model: Optional[Any] = None,
    max_events_for_embedding: int = 15000,
) -> Set[Tuple[str, str]]:
    """
    Find long-shot event pairs using double sliding windows moving in opposite
    directions. Only pairs with thematic similarity >= similarity_threshold
    are returned (e.g. 0.95 to keep calls minimum).

    - Left window slides forward: [0, W), [step, step+W), ...
    - Right window slides backward: [N-W, N), [N-W-step, N-step), ...
    - Compare events from left window with events from right window; keep
      (cause_id, effect_id) where cause is in left, effect in right, and
      similarity >= threshold.
    """
    pairs: Set[Tuple[str, str]] = set()
    if not _EMBED_AVAILABLE or not events:
        return pairs

    n = len(events)
    if n <= window_size * 2:
        return pairs

    if model is None:
        model = _get_embedding_model()
    if model is None:
        return pairs

    # Optionally cap to avoid memory blow-up
    if n > max_events_for_embedding:
        step = max(step, (n - 2 * window_size) // (max_events_for_embedding // (window_size * 2)))

    descriptions = [e.raw_description[:200] for e in events]
    embeddings = model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)

    left_start = 0
    right_end = n
    iteration = 0
    while left_start + window_size < right_end - window_size:
        left_end = min(left_start + window_size, n)
        right_start = max(right_end - window_size, 0)
        if left_end >= right_start:
            break

        left_emb = embeddings[left_start:left_end]   # (L, D)
        right_emb = embeddings[right_start:right_end] # (R, D)
        sim = np.dot(left_emb, right_emb.T)
        norms_left = np.linalg.norm(left_emb, axis=1, keepdims=True)
        norms_right = np.linalg.norm(right_emb, axis=1, keepdims=True)
        norms_left[norms_left == 0] = 1e-9
        norms_right[norms_right == 0] = 1e-9
        sim = sim / (norms_left * norms_right.T)

        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                if sim[i, j] >= similarity_threshold:
                    cause_idx = left_start + i
                    effect_idx = right_start + j
                    if cause_idx >= effect_idx:
                        continue
                    cause_id = events[cause_idx].id
                    effect_id = events[effect_idx].id
                    if cause_id in event_map and effect_id in event_map:
                        pairs.add((cause_id, effect_id))

        left_start += step
        right_end -= step
        iteration += 1
        if iteration % 10 == 0 and iteration > 0:
            print(f"[dynamic_context] Double sliding: {len(pairs):,} long-shot pairs so far...")

    return pairs


def get_local_and_scene_pairs(
    events: List[Any],
    scenes: Optional[List[Any]] = None,
    local_window: int = LOCAL_WINDOW_SIZE,
) -> Set[Tuple[str, str]]:
    """
    Pairs that are always checked for links:
    - Chronologically adjacent (i, i+1).
    - Within local_window distance (e.g. 5).
    - If scenes are provided, all pairs within the same scene (included_event_ids).
    """
    pairs: Set[Tuple[str, str]] = set()
    event_map = {e.id: e for e in events}

    # Consecutive and within local window
    for i, ev in enumerate(events):
        cause_id = ev.id
        cause_seq = ev.sequence
        for j in range(i + 1, min(i + 1 + local_window, len(events))):
            effect = events[j]
            if effect.sequence <= cause_seq:
                continue
            pairs.add((cause_id, effect.id))

    # Same-scene pairs (when scenes available)
    if scenes:
        for scene in scenes:
            eids = getattr(scene, "included_event_ids", None) or []
            for i, eid1 in enumerate(eids):
                if eid1 not in event_map:
                    continue
                for eid2 in eids[i + 1:]:
                    if eid2 not in event_map:
                        continue
                    ev1, ev2 = event_map[eid1], event_map[eid2]
                    if ev1.sequence < ev2.sequence:
                        pairs.add((eid1, eid2))
                    else:
                        pairs.add((eid2, eid1))

    return pairs


def get_dynamic_context_candidate_pairs(
    events: List[Any],
    entity_occurrences: Dict[str, List[Tuple[str, int]]],
    scenes: Optional[List[Any]] = None,
    thematic_threshold: float = THEMATIC_SIMILARITY_THRESHOLD,
    local_window: int = LOCAL_WINDOW_SIZE,
    double_window_size: int = DOUBLE_WINDOW_SIZE,
    double_window_step: int = DOUBLE_WINDOW_STEP,
    max_pairs: int = 50000,
    use_entity_guided: bool = True,
) -> List[Tuple[str, str]]:
    """
    Main entry: combine dynamic context (double sliding + 0.95 thematic) with
    local/scene pairs. Optionally add entity-guided pairs for high-precision
    links. Returns list of (cause_id, effect_id) for the remote LLM to label.

    - Long-shot: double sliding window + local embeddings, keep only >= thematic_threshold.
    - Local: adjacent and within local_window (and same-scene if scenes given).
    - Entity-guided (optional): same-entity consecutive appearances.
    """
    if not events:
        return []

    event_map = {e.id: e for e in events}
    print(f"[dynamic_context] Building candidate pairs for {len(events):,} events "
          f"(thematic >= {thematic_threshold}, local_window={local_window})")

    # 1) Local and scene pairs (always include)
    local_pairs = get_local_and_scene_pairs(events, scenes=scenes, local_window=local_window)
    print(f"[dynamic_context] Local/scene pairs: {len(local_pairs):,}")

    # 2) Long-shot pairs via double sliding window (local thematic engine as calculator)
    long_shot_pairs: Set[Tuple[str, str]] = set()
    if _EMBED_AVAILABLE and len(events) > double_window_size * 2:
        model = _get_embedding_model()
        if model:
            long_shot_pairs = get_long_shot_pairs_double_sliding(
                events,
                event_map,
                similarity_threshold=thematic_threshold,
                window_size=double_window_size,
                step=double_window_step,
                model=model,
            )
            print(f"[dynamic_context] Long-shot pairs (thematic >= {thematic_threshold}): {len(long_shot_pairs):,}")
    else:
        if not _EMBED_AVAILABLE:
            print("[dynamic_context] Embeddings not available; skipping long-shot thematic pairs.")
        elif len(events) <= double_window_size * 2:
            print("[dynamic_context] Too few events for double sliding; skipping long-shot.")

    # 3) Optional entity-guided (consecutive entity appearances)
    entity_pairs: Set[Tuple[str, str]] = set()
    if use_entity_guided and entity_occurrences:
        for entity_key, occurrences in entity_occurrences.items():
            if "place:" in entity_key:
                continue
            for i in range(len(occurrences) - 1):
                c_id, _ = occurrences[i]
                e_id, _ = occurrences[i + 1]
                if c_id in event_map and e_id in event_map:
                    entity_pairs.add((c_id, e_id))
        print(f"[dynamic_context] Entity-guided pairs: {len(entity_pairs):,}")

    # Merge and cap
    all_pairs = local_pairs | long_shot_pairs | entity_pairs
    all_list = list(all_pairs)
    if len(all_list) > max_pairs:
        local_list = list(local_pairs)
        long_shot_list = list(long_shot_pairs - local_pairs)
        entity_list = list(entity_pairs - local_pairs - long_shot_pairs)
        all_list = local_list
        remaining = max_pairs - len(all_list)
        if remaining > 0 and long_shot_list:
            take = min(remaining, len(long_shot_list))
            all_list.extend(long_shot_list[:take])
            remaining -= take
        if remaining > 0 and entity_list:
            take = min(remaining, len(entity_list))
            all_list.extend(entity_list[:take])
        print(f"[dynamic_context] Capped to {len(all_list):,} pairs (max_pairs={max_pairs})")

    return all_list
