"""
Causal Linking Module
Assesses candidate event pairs for causal relationships via LLM.
Semantic/thematic edges are now derived from theme annotations (see theme_annotation.py).
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any

from .schemas import CausalLink
from .utils import _hash_for_cache
from .llm_service import assessment_cache

# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------

PROMPT_CAUSAL_ASSESSMENT = """Analyze {count} event pairs for causal relationships using causal inference.

Causal reasoning: Use counterfactuals ("Would the effect have occurred without the cause?") and hypotheticals ("If the first event had not happened, would the second still have happened?"). Only label as causal when there is clear necessary or sufficient dependence in the narrative.

Available Causal Relations (use the *most specific* type that fits — do NOT default to generic enabling/precedence types when a specific psychological, social, or moral relation applies):
{causal_relations}

Selection guidance:
- Prefer specific types like EMOTIONAL_TRIGGER, MORAL_CHALLENGE, IDENTITY_CONFLICT,
  PERSONAL_TRANSFORMATION, REVEALS, EXPOSES, FORESHADOWS, BLOCKS, COMPLICATES,
  CAUSES_REVERSAL, REDEEMS, PERSUASION_ATTEMPT, FAMILY_INFLUENCE,
  INHERITED_OBLIGATION, DESIRE_OBSTRUCTION, MISSION_FAILURE, CONSCIENCE_CONFLICT,
  PERCEPTION_SHIFT — when the cause matches that specific mechanism.
- Use EVENT_ENABLES_NEXT only when the link is purely sequential enablement and
  no more specific psychological / social / moral / epistemic mechanism applies.
- Use DIRECT_CAUSE only when the cause is the proximate efficient cause and no
  more descriptive type fits.
- Do NOT pick the same relation type for every pair in this batch when they
  differ in mechanism — variety should reflect real differences in how the
  cause operates on the effect.

Pairs (cause → effect):
{pairs}

Return JSON:
{{
  "results": [
    {{
      "index": 1,
      "relationType": "EMOTIONAL_TRIGGER",
      "mechanism": "Estella's contempt produced shame about Pip's hands",
      "confidence": 0.9
    }},
    {{
      "index": 2,
      "relationType": "NONE",
      "mechanism": "",
      "confidence": 0.0
    }}
  ]
}}

Rules:
1. relationType: From list above or "NONE" (use NONE when no counterfactual dependence)
2. If no clear causal link (effect would have happened anyway), use "NONE"
3. confidence: 0.0 to 1.0 — high only when the mechanism is explicit in the text
4. If narrative context is provided for a pair, use it to ground your reasoning in the actual text.
5. mechanism: <=15 words, name the *specific* mechanism, not a restatement of the cause.

JSON only:"""

# ---------------------------------------------------------------------------
# CORE FUNCTION: Causal Assessment
# ---------------------------------------------------------------------------

async def assess_pairs_causal(
    pairs_batch: List[tuple],
    model: str,
    client: Any,
    causal_relations: List[str],
    llm_call_function: Any,
    passage_index: Optional[Any] = None,
) -> List[Optional[Dict]]:
    """
    Assesses a batch of pairs for causal links in a single API call.
    If passage_index is provided, retrieved narrative context is injected
    per pair to ground the LLM's causal reasoning in the actual text.
    Returns a list of result dicts (or None) aligned with pairs_batch.
    """
    if not pairs_batch:
        return []

    use_rag = passage_index is not None and getattr(passage_index, "is_ready", False)

    pairs_text_lines = []
    for i, (c_text, e_text, _, _) in enumerate(pairs_batch, 1):
        c_short = c_text[:80].replace("\n", " ")
        e_short = e_text[:80].replace("\n", " ")
        line = f"{i}. [{c_short}] -> [{e_short}]"
        if use_rag:
            passages = passage_index.retrieve(f"{c_text} {e_text}", top_k=2)
            if passages:
                context = " | ".join(passages)
                line += f"\n   Narrative context: \"{context[:250]}\""
        pairs_text_lines.append(line)

    pairs_block = "\n".join(pairs_text_lines)
    # Pass the full ontology so the LLM can pick from the specific relation types
    # rather than always falling back to the first 15 (which biased the
    # distribution toward EVENT_ENABLES_NEXT etc — see 0326 feedback).
    causal_str = ", ".join(causal_relations)

    prompt = PROMPT_CAUSAL_ASSESSMENT.format(
        count=len(pairs_batch),
        causal_relations=causal_str,
        pairs=pairs_block
    )

    # Cache key MUST hash the actual prompt content. The previous key only
    # hashed (prefix, batch-size, first-50-chars-of-ontology), which collides
    # across every 50-pair batch — so the first batch's response was silently
    # reused for every subsequent batch, randomizing relationType/mechanism
    # across the dataset (visible as a near-uniform ~498-per-type distribution
    # in gpt5_full.json).
    prefix = "causal_rag_v3" if use_rag else "causal_v3"
    cache_key = _hash_for_cache(f"{prefix}:{prompt}", model)

    try:
        data, _ = await llm_call_function(
            prompt, model, client, assessment_cache, cache_key, max_tokens=16000
        )

        if isinstance(data, dict) and "results" in data:
            results_map = {
                item.get("index"): item
                for item in data["results"]
                if item.get("index") is not None
            }
            return [results_map.get(i) for i in range(1, len(pairs_batch) + 1)]
        else:
            return [None] * len(pairs_batch)

    except Exception as e:
        print(f"[error] Causal assessment failed: {e}")
        return [None] * len(pairs_batch)


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

async def process_pairs_causal_only(
    pairs_with_text: List[Tuple[str, str, str, str]],
    model: str,
    client: Any,
    relation_ontology: List[str],
    theory_name: str,
    dag_validator: Any,
    ontology_validator: Any,
    llm_call_function: Any,
    max_concurrent_calls: int = 10,
    bulk_size: int = 50,
    passage_index: Optional[Any] = None,
) -> List[CausalLink]:
    """
    Main entry point called by pipeline.py.
    Returns causal links only — thematic links are built separately from theme annotations.
    """
    print(f"\n[{theory_name}] Processing {len(pairs_with_text)} pairs (causal only)...")

    all_causal = []

    for i in range(0, len(pairs_with_text), bulk_size * max_concurrent_calls):
        batch_end = min(i + bulk_size * max_concurrent_calls, len(pairs_with_text))

        chunks = []
        chunk_indices = []

        for j in range(i, batch_end, bulk_size):
            chunk = pairs_with_text[j:j + bulk_size]
            chunks.append(assess_pairs_causal(
                chunk, model, client, relation_ontology, llm_call_function,
                passage_index=passage_index,
            ))
            chunk_indices.append((j, j + len(chunk)))

        chunk_results = await asyncio.gather(*chunks)

        for result_list, (start_idx, _) in zip(chunk_results, chunk_indices):
            for k, c_res in enumerate(result_list):
                idx = start_idx + k
                if idx >= len(pairs_with_text):
                    break

                _, _, cause_id, effect_id = pairs_with_text[idx]

                if not c_res:
                    continue

                rel_type = str(c_res.get("relationType", "")).upper()
                if not rel_type or rel_type in ["NONE", "NULL"]:
                    continue

                if not ontology_validator.validate_relation_type(rel_type, theory_name):
                    continue

                direction = ontology_validator.get_relation_directionality(rel_type, theory_name)

                if dag_validator.add_edge(cause_id, effect_id):
                    all_causal.append(CausalLink(
                        source_event_id=cause_id,
                        target_event_id=effect_id,
                        relation_type=rel_type,
                        mechanism=c_res.get("mechanism", ""),
                        weight=float(c_res.get("weight", 0)),
                        confidence=float(c_res.get("confidence", 0)),
                        theory=f"@{theory_name.title()}",
                        directionality=direction
                    ))

        if (i // (bulk_size * max_concurrent_calls)) % 5 == 0:
            pct = 100 * (i / len(pairs_with_text))
            print(f"[progress] Found {len(all_causal)} causal links so far ({pct:.0f}%)...")

    print(f"[{theory_name}] Final: {len(all_causal)} causal links.")
    return all_causal