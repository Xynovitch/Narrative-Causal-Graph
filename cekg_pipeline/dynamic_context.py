"""
Dynamic context windows for causal linking.

Uses the local thematic engine (embeddings) as a calculation machine to find
high-similarity event pairs — both within scenes and long-range — instead of
proximity-only windows. The same 0.95 threshold is applied to both pools so
the LLM only assesses pairs with genuine thematic overlap.

Pipeline:
1. Adjacent pairs (i, i+1) — always included, unconditional baseline.
2. Scene pairs — within each scene, all pairs filtered by >= threshold similarity.
3. Long-shot pairs — double sliding window (left forward, right backward),
   pairs kept only if similarity >= threshold.
4. Entity-guided pairs — consecutive same-entity appearances.
5. Cap with even split between scene-filtered and long-shot pools.
"""

from typing import List, Tuple, Set, Dict, Optional, Any

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBED_AVAILABLE = True
except ImportError:
    _EMBED_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# Defaults
THEMATIC_SIMILARITY_THRESHOLD = 0.95
BM25_TOP_K = 5   # top-K BM25 matches per event
LOCAL_WINDOW_SIZE = 5        # fallback window when embeddings unavailable
DOUBLE_WINDOW_SIZE = 100     # events per sliding window (long-shot)
DOUBLE_WINDOW_STEP = 50


def _get_embedding_model() -> Optional[Any]:
    if not _EMBED_AVAILABLE:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _encode(events: List[Any], model: Any) -> "np.ndarray":
    descriptions = [e.raw_description[:200] for e in events]
    return model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)


def _cosine_sim(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Cosine similarity between rows of a and rows of b. Returns (len(a), len(b))."""
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    norms_a[norms_a == 0] = 1e-9
    norms_b[norms_b == 0] = 1e-9
    return np.dot(a / norms_a, (b / norms_b).T)


def get_adjacent_pairs(events: List[Any]) -> Set[Tuple[str, str]]:
    """Always-include baseline: every consecutive (i, i+1) pair."""
    pairs: Set[Tuple[str, str]] = set()
    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if a.sequence < b.sequence:
            pairs.add((a.id, b.id))
    return pairs


def get_scene_pairs_by_similarity(
    events: List[Any],
    scenes: List[Any],
    embeddings: "np.ndarray",
    event_index: Dict[str, int],
    similarity_threshold: float = THEMATIC_SIMILARITY_THRESHOLD,
) -> Set[Tuple[str, str]]:
    """
    For each scene, compute pairwise cosine similarity between all events in
    that scene and keep only pairs with similarity >= threshold.

    Uses precomputed embeddings (indexed by event_index) so embeddings are
    only computed once across the whole pipeline.
    """
    pairs: Set[Tuple[str, str]] = set()
    event_map = {e.id: e for e in events}

    for scene in scenes:
        eids = getattr(scene, "included_event_ids", None) or []
        scene_events = sorted(
            [event_map[eid] for eid in eids if eid in event_map],
            key=lambda e: e.sequence,
        )
        if len(scene_events) < 2:
            continue

        idxs = [event_index[e.id] for e in scene_events if e.id in event_index]
        if len(idxs) < 2:
            continue

        emb = embeddings[idxs]          # (K, D)
        sim = _cosine_sim(emb, emb)     # (K, K)

        for i in range(len(scene_events)):
            for j in range(i + 1, len(scene_events)):
                if sim[i, j] >= similarity_threshold:
                    ev1, ev2 = scene_events[i], scene_events[j]
                    if ev1.sequence < ev2.sequence:
                        pairs.add((ev1.id, ev2.id))
                    else:
                        pairs.add((ev2.id, ev1.id))

    return pairs


def get_long_shot_pairs_double_sliding(
    events: List[Any],
    event_map: Dict[str, Any],
    embeddings: "np.ndarray",
    similarity_threshold: float = THEMATIC_SIMILARITY_THRESHOLD,
    window_size: int = DOUBLE_WINDOW_SIZE,
    step: int = DOUBLE_WINDOW_STEP,
) -> Set[Tuple[str, str]]:
    """
    Find long-shot event pairs using double sliding windows moving in opposite
    directions. Left window slides forward, right window slides backward;
    only pairs with thematic similarity >= threshold are kept.
    Accepts precomputed embeddings to avoid recomputation.
    """
    pairs: Set[Tuple[str, str]] = set()
    n = len(events)
    if n <= window_size * 2:
        return pairs

    left_start = 0
    right_end = n
    iteration = 0
    while left_start + window_size < right_end - window_size:
        left_end = min(left_start + window_size, n)
        right_start = max(right_end - window_size, 0)
        if left_end >= right_start:
            break

        left_emb = embeddings[left_start:left_end]
        right_emb = embeddings[right_start:right_end]
        sim = _cosine_sim(left_emb, right_emb)

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
        if iteration % 10 == 0:
            print(f"[dynamic_context] Long-shot sliding: {len(pairs):,} pairs so far...")

    return pairs


def _get_local_fallback_pairs(
    events: List[Any],
    scenes: Optional[List[Any]],
    local_window: int,
) -> Set[Tuple[str, str]]:
    """Proximity-only fallback used when embeddings are unavailable."""
    pairs: Set[Tuple[str, str]] = set()
    event_map = {e.id: e for e in events}

    for i, ev in enumerate(events):
        for j in range(i + 1, min(i + 1 + local_window, len(events))):
            effect = events[j]
            if effect.sequence > ev.sequence:
                pairs.add((ev.id, effect.id))

    if scenes:
        for scene in scenes:
            eids = getattr(scene, "included_event_ids", None) or []
            scene_events = sorted(
                [event_map[eid] for eid in eids if eid in event_map],
                key=lambda e: e.sequence,
            )
            for i, ev1 in enumerate(scene_events):
                for ev2 in scene_events[i + 1: i + 1 + local_window]:
                    pairs.add((ev1.id, ev2.id))

    return pairs


def get_bm25_pairs(
    events: List[Any],
    top_k: int = BM25_TOP_K,
) -> Set[Tuple[str, str]]:
    """
    Keyword-based candidate discovery using BM25.
    For each event, retrieves the top-K events with highest keyword overlap.
    Complements cosine similarity for named entities and domain terms that
    dense embeddings tend to compress away.
    """
    if not _BM25_AVAILABLE or not _EMBED_AVAILABLE or not events:
        return set()

    corpus = [e.raw_description.lower().split() for e in events]
    bm25 = BM25Okapi(corpus)
    pairs: Set[Tuple[str, str]] = set()

    for i, ev in enumerate(events):
        scores = bm25.get_scores(corpus[i]).copy()
        scores[i] = -1.0   # exclude self-match
        top_indices = np.argsort(scores)[-top_k:]
        for j in top_indices:
            if scores[j] <= 0:
                continue
            a, b = (events[i], events[j]) if events[i].sequence <= events[j].sequence else (events[j], events[i])
            if a.sequence < b.sequence:
                pairs.add((a.id, b.id))

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
    Build candidate pairs for the remote LLM to label.

    With embeddings available:
      - Adjacent pairs: always included.
      - Scene pairs: within-scene pairs filtered by >= thematic_threshold.
      - Long-shot pairs: double sliding window filtered by >= thematic_threshold.
      - Entity-guided: consecutive same-entity appearances.
      - Cap: even split between scene-filtered and long-shot pools.

    Without embeddings:
      - Falls back to proximity window + entity-guided pairs.
    """
    if not events:
        return []

    event_map = {e.id: e for e in events}
    print(f"[dynamic_context] Building candidate pairs for {len(events):,} events "
          f"(thematic >= {thematic_threshold})")

    # --- 1. Adjacent pairs (unconditional baseline) ---
    adjacent_pairs = get_adjacent_pairs(events)
    print(f"[dynamic_context] Adjacent pairs: {len(adjacent_pairs):,}")

    # --- 2. Entity-guided pairs ---
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

    # --- 3. BM25 keyword pairs ---
    bm25_pairs: Set[Tuple[str, str]] = set()
    if _BM25_AVAILABLE:
        bm25_pairs = get_bm25_pairs(events)
        bm25_pairs = {p for p in bm25_pairs if p[0] in event_map and p[1] in event_map}
        print(f"[dynamic_context] BM25 keyword pairs: {len(bm25_pairs):,}")

    # --- 4. Similarity-filtered pools (scene + long-shot) ---
    scene_pairs: Set[Tuple[str, str]] = set()
    long_shot_pairs: Set[Tuple[str, str]] = set()

    if _EMBED_AVAILABLE:
        embed_model = _get_embedding_model()
        if embed_model is not None:
            print(f"[dynamic_context] Encoding {len(events):,} events...")
            embeddings = _encode(events, embed_model)
            event_index = {e.id: i for i, e in enumerate(events)}

            # Scene pairs with similarity filter
            if scenes:
                scene_pairs = get_scene_pairs_by_similarity(
                    events, scenes, embeddings, event_index, thematic_threshold
                )
                print(f"[dynamic_context] Scene pairs (sim >= {thematic_threshold}): {len(scene_pairs):,}")

            # Long-shot pairs with similarity filter
            if len(events) > double_window_size * 2:
                long_shot_pairs = get_long_shot_pairs_double_sliding(
                    events, event_map, embeddings,
                    similarity_threshold=thematic_threshold,
                    window_size=double_window_size,
                    step=double_window_step,
                )
                print(f"[dynamic_context] Long-shot pairs (sim >= {thematic_threshold}): {len(long_shot_pairs):,}")
        else:
            print("[dynamic_context] Embedding model unavailable; using proximity fallback.")
    else:
        print("[dynamic_context] sentence-transformers not available; using proximity fallback.")

    # Fallback when embeddings failed
    if not scene_pairs and not long_shot_pairs:
        fallback = _get_local_fallback_pairs(events, scenes, local_window)
        print(f"[dynamic_context] Fallback local/scene pairs: {len(fallback):,}")
        all_pairs = adjacent_pairs | entity_pairs | bm25_pairs | fallback
        all_list = list(all_pairs)
        if len(all_list) > max_pairs:
            all_list = all_list[:max_pairs]
            print(f"[dynamic_context] Capped to {len(all_list):,} pairs (max_pairs={max_pairs})")
        return all_list

    # --- 5. Merge all pools ---
    all_pairs = adjacent_pairs | entity_pairs | bm25_pairs | scene_pairs | long_shot_pairs

    # Safety ceiling only
    if len(all_pairs) > max_pairs:
        print(f"[dynamic_context] Safety cap: {len(all_pairs):,} → {max_pairs:,} "
              f"(lower threshold or raise --max-pairs to avoid this)")
        # Priority: scene-filtered > long-shot > BM25 > entity > adjacent
        result: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for pool in (scene_pairs, long_shot_pairs, bm25_pairs, entity_pairs, adjacent_pairs):
            for p in pool:
                if len(result) >= max_pairs:
                    break
                if p not in seen:
                    result.append(p)
                    seen.add(p)
        all_pairs = set(result)

    all_list = list(all_pairs)
    bm25_only = len(bm25_pairs - adjacent_pairs - entity_pairs)
    print(f"[dynamic_context] Selected {len(all_list):,} pairs "
          f"(adj={len(adjacent_pairs):,}, entity={len(entity_pairs - adjacent_pairs):,}, "
          f"bm25={bm25_only:,}, "
          f"scene={len(scene_pairs - adjacent_pairs - entity_pairs - bm25_pairs):,}, "
          f"long-shot={len(long_shot_pairs - adjacent_pairs - entity_pairs - bm25_pairs - scene_pairs):,})")
    return all_list
