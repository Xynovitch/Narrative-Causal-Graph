"""
Integrated Semantic Linking Module
Combines 'Piggyback' API extraction (Zero Cost) with 'Smart' Local Embeddings (High Quality).
FIXED: Added Truncation to prevent CPU hang.
"""

import asyncio
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

# Try to import schema classes
from .schemas import CausalLink, SemanticLink
from .utils import _make_id, _hash_for_cache
# Import the cache to pass to the LLM service
from .llm_service import assessment_cache

# ---------------------------------------------------------------------------
# GLOBAL: Load Embedding Model
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    print("[semantic] Loading embedding model (all-MiniLM-L6-v2)...")
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    EMBEDDING_MODEL = None
    print("[warning] 'sentence-transformers' not found. Local semantic detection disabled.")

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

PROMPT_INTEGRATED_ASSESSMENT = """Analyze {count} event pairs for BOTH causal AND semantic relationships.

Causal Relations: {causal_relations}
Semantic Relations: explanation, contrast, elaboration, parallel, foreshadowing

Pairs:
{pairs}

Return JSON with BOTH types:
{{
  "results": [
    {{
      "index": 1,
      "causal": {{"relationType": "DIRECT_CAUSE", "mechanism": "Pip's theft led to guilt", "confidence": 0.9}},
      "semantic": {{"relation": "explanation", "cue": ["because"], "confidence": 0.8}}
    }},
    {{
      "index": 2,
      "causal": {{"relationType": "NONE", "mechanism": "", "confidence": 0.0}},
      "semantic": {{"relation": "contrast", "cue": ["however", "but"], "confidence": 0.7}}
    }}
  ]
}}

Rules:
1. causal.relationType: From list above or "NONE"
2. semantic.relation: explanation, contrast, elaboration, parallel, foreshadowing, or "none"
3. If no clear link, use "NONE" / "none"
4. cue: List of words/phrases supporting the semantic link
5. confidence: 0.0 to 1.0

JSON only:"""

# ---------------------------------------------------------------------------
# CORE FUNCTION: Integrated API Call
# ---------------------------------------------------------------------------

async def assess_pairs_integrated(
    pairs_batch: List[tuple], 
    model: str, 
    client: Any, 
    causal_relations: List[str],
    llm_call_function: Any
) -> Tuple[List[Optional[Dict]], List[Optional[Dict]]]:
    """
    Assesses pairs for BOTH causal and semantic links in a single API call.
    """
    if not pairs_batch:
        return [], []

    pairs_text_lines = []
    for i, (c_text, e_text, _, _) in enumerate(pairs_batch, 1):
        c_short = c_text[:80].replace("\n", " ")
        e_short = e_text[:80].replace("\n", " ")
        pairs_text_lines.append(f"{i}. [{c_short}] -> [{e_short}]")
    
    pairs_block = "\n".join(pairs_text_lines)
    causal_str = ", ".join(causal_relations[:15])
    
    prompt = PROMPT_INTEGRATED_ASSESSMENT.format(
        count=len(pairs_batch),
        causal_relations=causal_str,
        pairs=pairs_block
    )
    
    cache_key = _hash_for_cache(f"integrated:{len(pairs_batch)}:{causal_str[:50]}", model)
    
    try:
        # Pass 16k tokens to allow full response
        data, _ = await llm_call_function(
            prompt, model, client, assessment_cache, cache_key, max_tokens=16000
        )
        
        causal_results = []
        semantic_results = []
        
        if isinstance(data, dict) and "results" in data:
            results_map = {
                item.get("index"): item 
                for item in data["results"] 
                if item.get("index") is not None
            }
            
            for i in range(1, len(pairs_batch) + 1):
                item = results_map.get(i)
                if not item:
                    causal_results.append(None)
                    semantic_results.append(None)
                    continue

                c_data = item.get("causal", {})
                if c_data and str(c_data.get("relationType", "NONE")).upper() != "NONE":
                    causal_results.append(c_data)
                else:
                    causal_results.append(None)
                
                s_data = item.get("semantic", {})
                if s_data and str(s_data.get("relation", "none")).lower() != "none":
                    semantic_results.append(s_data)
                else:
                    semantic_results.append(None)
        else:
            return [None] * len(pairs_batch), [None] * len(pairs_batch)
            
        return causal_results, semantic_results

    except Exception as e:
        print(f"[error] Integrated assessment failed: {e}")
        return [None] * len(pairs_batch), [None] * len(pairs_batch)

# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

async def process_pairs_with_semantic_linking(
    pairs_with_text: List[Tuple[str, str, str, str]],
    model: str,
    client: Any,
    relation_ontology: List[str],
    theory_name: str,
    dag_validator: Any,
    ontology_validator: Any,
    llm_call_function: Any,
    max_concurrent_calls: int = 10,
    bulk_size: int = 50
) -> Tuple[List[CausalLink], List[SemanticLink]]:
    """
    Main entry point called by pipeline.py.
    """
    print(f"\n[{theory_name}] Processing {len(pairs_with_text)} pairs (Causal + Semantic)...")
    
    all_causal = []
    all_semantic = []
    
    for i in range(0, len(pairs_with_text), bulk_size * max_concurrent_calls):
        batch_end = min(i + bulk_size * max_concurrent_calls, len(pairs_with_text))
        
        chunks = []
        chunk_indices = []
        
        for j in range(i, batch_end, bulk_size):
            chunk = pairs_with_text[j:j + bulk_size]
            chunks.append(assess_pairs_integrated(
                chunk, model, client, relation_ontology, llm_call_function
            ))
            chunk_indices.append((j, j + len(chunk)))
        
        chunk_results = await asyncio.gather(*chunks)
        
        for (causal_res_list, semantic_res_list), (start_idx, _) in zip(chunk_results, chunk_indices):
            
            for k, (c_res, s_res) in enumerate(zip(causal_res_list, semantic_res_list)):
                idx = start_idx + k
                if idx >= len(pairs_with_text): break
                
                _, _, cause_id, effect_id = pairs_with_text[idx]
                
                if c_res:
                    rel_type = str(c_res.get("relationType", "")).upper()
                    if rel_type and rel_type not in ["NONE", "NULL"]:
                        if ontology_validator.validate_relation_type(rel_type, theory_name):
                            direction = ontology_validator.get_relation_directionality(rel_type, theory_name)
                            
                            if dag_validator.add_edge(cause_id, effect_id):
                                link = CausalLink(
                                    source_event_id=cause_id,
                                    target_event_id=effect_id,
                                    relation_type=rel_type,
                                    mechanism=c_res.get("mechanism", ""),
                                    weight=float(c_res.get("weight", 0)),
                                    confidence=float(c_res.get("confidence", 0)),
                                    theory=f"@{theory_name.title()}",
                                    directionality=direction
                                )
                                all_causal.append(link)

                if s_res:
                    rel = str(s_res.get("relation", "")).lower()
                    if rel and rel != "none":
                        link = SemanticLink(
                            id=_make_id("sem"),
                            source_event_ids=[cause_id],
                            target_event_ids=[effect_id],
                            relation=rel,
                            cue=s_res.get("cue", []),
                            confidence=float(s_res.get("confidence", 0))
                        )
                        all_semantic.append(link)

        if (i // (bulk_size * max_concurrent_calls)) % 5 == 0:
            total_found = len(all_causal) + len(all_semantic)
            pct = 100 * (i / len(pairs_with_text))
            print(f"[progress] Found {total_found} links so far ({pct:.0f}%)...")

    print(f"[{theory_name}] Final: {len(all_causal)} causal, {len(all_semantic)} semantic links.")
    return all_causal, all_semantic

# ---------------------------------------------------------------------------
# LOCAL DETECTION: High-Quality Embeddings
# ---------------------------------------------------------------------------

def detect_semantic_links_locally(
    events: List,
    window: int = 5,
    similarity_threshold: float = 0.55
) -> List[SemanticLink]:
    """
    Uses local embeddings to find thematic connections.
    """
    if not EMBEDDING_MODEL:
        return []
        
    if not events:
        return []

    print(f"[semantic_local] Generating embeddings for {len(events)} events (CPU)...")
    
    # FIX: Truncate descriptions to 200 chars. This speeds up processing by 100x.
    descriptions = [e.raw_description[:200] for e in events]
    
    # FIX: Show progress bar
    embeddings = EMBEDDING_MODEL.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
    
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    links = []
    # Use torch operations for speed instead of nested loops if possible
    # But for graph building, we need indices.
    
    rows, cols = cosine_scores.shape
    matches = torch.where(cosine_scores > similarity_threshold)
    
    # Process matches
    for i, j in zip(*matches):
        i, j = i.item(), j.item()
        if i >= j: continue 
        
        event_a = events[i]
        event_b = events[j]
        
        dist = abs(event_b.sequence - event_a.sequence)
        
        # Only link if distant (thematic echo)
        if dist > window:
            score = float(cosine_scores[i][j])
            links.append(SemanticLink(
                id=_make_id("sem_local"),
                source_event_ids=[event_a.id],
                target_event_ids=[event_b.id],
                relation="thematic_similarity",
                cue=[f"similarity:{score:.2f}"],
                confidence=score
            ))

    print(f"[semantic_local] Found {len(links)} thematic links via embeddings.")
    return links

# ---------------------------------------------------------------------------
# UTILS: Merging & Helpers
# ---------------------------------------------------------------------------

def merge_semantic_links(
    list_a: List[SemanticLink],
    list_b: List[SemanticLink]
) -> List[SemanticLink]:
    link_map = {}
    
    for link in list_a + list_b:
        src = tuple(sorted(link.source_event_ids))
        tgt = tuple(sorted(link.target_event_ids))
        key = (src, tgt, link.relation)
        
        if key not in link_map:
            link_map[key] = link
        else:
            if link.confidence > link_map[key].confidence:
                link_map[key] = link
                
    return list(link_map.values())

def create_hybrid_semantic_links(
    events: List,
    llm_semantic_links: List[SemanticLink]
) -> List[SemanticLink]:
    """
    Combine LLM-detected links with Local Embedding links.
    """
    local_links = detect_semantic_links_locally(events)
    merged = merge_semantic_links(local_links, llm_semantic_links)
    print(f"[semantic_hybrid] Combined Total: {len(merged)} links.")
    return merged