"""
Integrated Semantic Linking Module
Combines 'Piggyback' API extraction (Zero Cost) with 'Smart' Local Embeddings (High Quality).
"""

import asyncio
import re
import json
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

# Try to import schema classes; adapt if your project structure is different
from .schemas import CausalLink, SemanticLink
from .utils import _make_id, _hash_for_cache

# ---------------------------------------------------------------------------
# GLOBAL: Load Embedding Model (The "Smart" Brain)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util
    # Load the model once when the module is imported
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
# CORE FUNCTION: Integrated API Call (The "Piggyback")
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
    Zero additional cost.
    """
    if not pairs_batch:
        return [], []

    # 1. Format pairs for the prompt
    pairs_text_lines = []
    for i, (c_text, e_text, _, _) in enumerate(pairs_batch, 1):
        # Truncate descriptions to save tokens
        c_short = c_text[:80].replace("\n", " ")
        e_short = e_text[:80].replace("\n", " ")
        pairs_text_lines.append(f"{i}. [{c_short}] -> [{e_short}]")
    
    pairs_block = "\n".join(pairs_text_lines)
    causal_str = ", ".join(causal_relations[:15]) # Limit ontology size
    
    prompt = PROMPT_INTEGRATED_ASSESSMENT.format(
        count=len(pairs_batch),
        causal_relations=causal_str,
        pairs=pairs_block
    )
    
    # 2. Generate Cache Key
    # We include 'integrated' in the key so it doesn't clash with old causal-only calls
    cache_key = _hash_for_cache(f"integrated:{len(pairs_batch)}:{causal_str[:50]}", model)
    
    try:
        # 3. Call LLM (using the wrapper passed from pipeline)
        data, _ = await llm_call_function(
            prompt, model, client, cache_key, max_tokens=2048
        )
        
        # 4. Parse Results
        causal_results = []
        semantic_results = []
        
        if isinstance(data, dict) and "results" in data:
            # Map results by index to handle potential LLM reordering
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

                # Extract Causal
                c_data = item.get("causal", {})
                if c_data and str(c_data.get("relationType", "NONE")).upper() != "NONE":
                    causal_results.append(c_data)
                else:
                    causal_results.append(None)
                
                # Extract Semantic
                s_data = item.get("semantic", {})
                if s_data and str(s_data.get("relation", "none")).lower() != "none":
                    semantic_results.append(s_data)
                else:
                    semantic_results.append(None)
        else:
            # Fallback for malformed JSON
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
    causal_relations: List[str],
    theory_name: str,
    dag_validator: Any,
    ontology_validator: Any,
    llm_call_function: Any,
    max_concurrent_calls: int = 10,
    bulk_size: int = 50
) -> Tuple[List[CausalLink], List[SemanticLink]]:
    """
    Main entry point called by pipeline.py.
    Orchestrates the batch processing and object creation.
    """
    print(f"\n[{theory_name}] Processing {len(pairs_with_text)} pairs (Causal + Semantic)...")
    
    all_causal = []
    all_semantic = []
    
    # Loop through data in chunks
    for i in range(0, len(pairs_with_text), bulk_size * max_concurrent_calls):
        batch_end = min(i + bulk_size * max_concurrent_calls, len(pairs_with_text))
        
        # Prepare concurrent tasks
        chunks = []
        chunk_indices = []
        
        for j in range(i, batch_end, bulk_size):
            chunk = pairs_with_text[j:j + bulk_size]
            chunks.append(assess_pairs_integrated(
                chunk, model, client, causal_relations, llm_call_function
            ))
            chunk_indices.append((j, j + len(chunk)))
        
        # Run tasks
        chunk_results = await asyncio.gather(*chunks)
        
        # Process results
        for (causal_res_list, semantic_res_list), (start_idx, _) in zip(chunk_results, chunk_indices):
            
            for k, (c_res, s_res) in enumerate(zip(causal_res_list, semantic_res_list)):
                idx = start_idx + k
                if idx >= len(pairs_with_text): break
                
                _, _, cause_id, effect_id = pairs_with_text[idx]
                
                # A) Build Causal Link
                if c_res:
                    rel_type = str(c_res.get("relationType", "")).upper()
                    if rel_type and rel_type not in ["NONE", "NULL"]:
                        # Validate against ontology
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

                # B) Build Semantic Link
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

        # Progress bar
        if (i // (bulk_size * max_concurrent_calls)) % 5 == 0:
            total_found = len(all_causal) + len(all_semantic)
            pct = 100 * (i / len(pairs_with_text))
            print(f"[progress] Found {total_found} links so far ({pct:.0f}%)...")

    print(f"[{theory_name}] Final: {len(all_causal)} causal, {len(all_semantic)} semantic links.")
    return all_causal, all_semantic

# ---------------------------------------------------------------------------
# LOCAL DETECTION: High-Quality Embeddings (The "Smart" Upgrade)
# ---------------------------------------------------------------------------

def detect_semantic_links_locally(
    events: List,
    window: int = 5,
    similarity_threshold: float = 0.55
) -> List[SemanticLink]:
    """
    Uses local embeddings to find thematic connections.
    Replaces the old Regex/keyword approach.
    """
    if not EMBEDDING_MODEL:
        return []
        
    if not events:
        return []

    print(f"[semantic_local] Generating embeddings for {len(events)} events (CPU)...")
    
    # 1. Vectorize
    # We use raw_description. If too short, maybe combine with name.
    descriptions = [e.raw_description for e in events]
    embeddings = EMBEDDING_MODEL.encode(descriptions, convert_to_tensor=True)
    
    # 2. Compute Similarity Matrix
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    links = []
    rows, cols = cosine_scores.shape
    
    # 3. Iterate and Filter
    # We only look at upper triangle (j > i) to avoid duplicates
    count = 0
    for i in range(rows):
        for j in range(i + 1, rows):
            score = float(cosine_scores[i][j])
            
            if score > similarity_threshold:
                event_a = events[i]
                event_b = events[j]
                
                # FILTER: Skip if they are immediate neighbors (likely just narrative flow)
                # We want *thematic* links that span across the text
                dist = abs(event_b.sequence - event_a.sequence)
                
                if dist > window:
                    links.append(SemanticLink(
                        id=_make_id("sem_local"),
                        source_event_ids=[event_a.id],
                        target_event_ids=[event_b.id],
                        relation="thematic_similarity",
                        cue=[f"similarity:{score:.2f}"],
                        confidence=score
                    ))
                    count += 1

    print(f"[semantic_local] Found {count} thematic links via embeddings.")
    return links

# ---------------------------------------------------------------------------
# UTILS: Merging & Helpers
# ---------------------------------------------------------------------------

def merge_semantic_links(
    list_a: List[SemanticLink],
    list_b: List[SemanticLink]
) -> List[SemanticLink]:
    """
    Merge two lists of semantic links, keeping the higher confidence one for duplicates.
    """
    link_map = {}
    
    for link in list_a + list_b:
        # Sort IDs to ensure A->B is same as B->A (if we treat semantic as undirected)
        # But SemanticLink has distinct source/target. We assume direction matters slightly.
        # Key = (tuple(source), tuple(target), relation)
        
        src = tuple(sorted(link.source_event_ids))
        tgt = tuple(sorted(link.target_event_ids))
        key = (src, tgt, link.relation)
        
        if key not in link_map:
            link_map[key] = link
        else:
            # Keep higher confidence
            if link.confidence > link_map[key].confidence:
                link_map[key] = link
                
    return list(link_map.values())

def create_hybrid_semantic_links(
    events: List,
    llm_semantic_links: List[SemanticLink]
) -> List[SemanticLink]:
    """
    Combine LLM-detected links (from API) with Local Embedding links (from CPU).
    """
    # 1. Get Local Links (Smart & Free)
    local_links = detect_semantic_links_locally(events)
    
    # 2. Merge
    merged = merge_semantic_links(local_links, llm_semantic_links)
    
    print(f"[semantic_hybrid] Combined Total: {len(merged)} links.")
    return merged