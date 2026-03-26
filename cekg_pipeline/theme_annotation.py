"""
Thematic Layer v2 — annotates events with structural literary theme participation.

Themes: POWER, WEALTH, KINSHIP, JUSTICE, KNOWLEDGE
Roles: initiating, enabling, constraining, mediating, escalating, resolving, revealing
Involvement: direct, indirect, latent, none
"""
import asyncio
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional

from .schemas import CEKEvent, CausalLink, Scene
from .llm_service import annotate_single_event_theme

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THEME_SET = {"POWER", "WEALTH", "KINSHIP", "JUSTICE", "KNOWLEDGE"}

ROLE_SET = {
    "initiating", "enabling", "constraining", "mediating",
    "escalating", "resolving", "revealing"
}

INVOLVEMENT_SET = {"direct", "indirect", "latent", "none"}

# ---------------------------------------------------------------------------
# Fine-grained relation → broad supertype mapping
# ---------------------------------------------------------------------------

FINE_TO_SUPERTYPE: Dict[str, str] = {
    # Causal production
    "CAUSES": "CAUSAL_PRODUCTION",
    "DIRECT_CAUSE": "CAUSAL_PRODUCTION",
    "ENABLES": "CAUSAL_PRODUCTION",
    "FACILITATES": "CAUSAL_PRODUCTION",
    "TRIGGERS": "CAUSAL_PRODUCTION",
    # Causal constraint
    "PREVENTS": "CAUSAL_CONSTRAINT",
    "BLOCKS": "CAUSAL_CONSTRAINT",
    "INHIBITS": "CAUSAL_CONSTRAINT",
    "COMPLICATES": "CAUSAL_CONSTRAINT",
    "OPPOSES": "CAUSAL_CONSTRAINT",
    # Revelation / epistemic
    "REVEALS": "REVELATION_EPISTEMIC",
    "EXPOSES": "REVELATION_EPISTEMIC",
    "CONCEALS": "REVELATION_EPISTEMIC",
    "FORESHADOWS": "REVELATION_EPISTEMIC",
    # Mediation / transfer
    "INFORMS": "MEDIATION_TRANSFER",
    "MEDIATES": "MEDIATION_TRANSFER",
    "TRANSFERS": "MEDIATION_TRANSFER",
    "DELEGATES": "MEDIATION_TRANSFER",
    # Narrative escalation
    "ESCALATES": "NARRATIVE_ESCALATION",
    "COMPLICATES_FURTHER": "NARRATIVE_ESCALATION",
    "CHALLENGES": "NARRATIVE_ESCALATION",
    "MORAL_CHALLENGE": "NARRATIVE_ESCALATION",
    # Narrative resolution
    "RESOLVES": "NARRATIVE_RESOLUTION",
    "CONCLUDES": "NARRATIVE_RESOLUTION",
    "REDEEMS": "NARRATIVE_RESOLUTION",
    # Thematic / semantic
    "CONTRASTS": "THEMATIC_CONTRAST",
    "MIRRORS": "THEMATIC_CONTRAST",
    "EXPLAINS": "THEMATIC_EXPLANATION",
    "SUPPORTS": "THEMATIC_EXPLANATION",
}


# ---------------------------------------------------------------------------
# assign_edge_supertypes
# ---------------------------------------------------------------------------

def assign_edge_supertypes(causal_links: List[CausalLink]) -> None:
    """Mutate each CausalLink in-place, setting edge_supertype from FINE_TO_SUPERTYPE."""
    for link in causal_links:
        rt = (link.relation_type or "").upper()
        link.edge_supertype = FINE_TO_SUPERTYPE.get(rt)


# ---------------------------------------------------------------------------
# attach_scene_ids_to_events
# ---------------------------------------------------------------------------

def attach_scene_ids_to_events(events: List[CEKEvent], scenes: List[Scene]) -> None:
    """Mutate each CEKEvent in-place, setting scene_id based on scene membership."""
    event_to_scene: Dict[str, str] = {}
    for scene in scenes:
        for eid in scene.included_event_ids:
            event_to_scene[eid] = scene.id

    for event in events:
        event.scene_id = event_to_scene.get(event.id)


# ---------------------------------------------------------------------------
# build_local_causal_context
# ---------------------------------------------------------------------------

def build_local_causal_context(
    event: CEKEvent,
    causes_by_target: Dict[str, List[CausalLink]],
    effects_by_source: Dict[str, List[CausalLink]],
    event_map: Dict[str, CEKEvent],
) -> Dict[str, Any]:
    """Build a dict describing an event and its immediate causal neighbourhood."""
    causes = causes_by_target.get(event.id, [])[:2]
    effects = effects_by_source.get(event.id, [])[:2]

    def _summarise(lnk: CausalLink, other_id: str) -> Dict[str, Any]:
        other = event_map.get(other_id)
        return {
            "event_id": other_id,
            "description": other.raw_description if other else "",
            "relation_type": lnk.relation_type,
            "mechanism": lnk.mechanism,
        }

    return {
        "event_id": event.id,
        "description": event.raw_description,
        "actors": event.actors,
        "patients": event.patients,
        "why_factors": event.why_factors,
        "scene_id": event.scene_id,
        "chapter": event.chapter,
        "immediate_causes": [_summarise(lnk, lnk.source_event_id) for lnk in causes],
        "immediate_effects": [_summarise(lnk, lnk.target_event_id) for lnk in effects],
    }


# ---------------------------------------------------------------------------
# apply_theme_bridge_rule
# ---------------------------------------------------------------------------

def apply_theme_bridge_rule(
    events: List[CEKEvent],
    causal_links: List[CausalLink]
) -> None:
    """
    Deterministic post-processing: if an event has involvement='none' for a theme
    but an adjacent cause or effect has involvement='direct', upgrade the event to
    involvement='indirect' and role='mediating'.
    """
    event_map: Dict[str, CEKEvent] = {e.id: e for e in events}

    # Build adjacency: event_id -> set of neighbour event_ids
    neighbours: Dict[str, List[str]] = {}
    for lnk in causal_links:
        neighbours.setdefault(lnk.source_event_id, []).append(lnk.target_event_id)
        neighbours.setdefault(lnk.target_event_id, []).append(lnk.source_event_id)

    for event in events:
        ann = event.theme_annotations
        if not ann:
            continue
        for theme in THEME_SET:
            theme_data = ann.get(theme, {})
            if not isinstance(theme_data, dict):
                continue
            if theme_data.get("involvement") != "none":
                continue
            # Check neighbours
            for nb_id in neighbours.get(event.id, []):
                nb = event_map.get(nb_id)
                if nb is None:
                    continue
                nb_theme_data = nb.theme_annotations.get(theme, {})
                if isinstance(nb_theme_data, dict) and nb_theme_data.get("involvement") == "direct":
                    theme_data["involvement"] = "indirect"
                    theme_data["role"] = "mediating"
                    break


# ---------------------------------------------------------------------------
# annotate_event_themes  (main entry point)
# ---------------------------------------------------------------------------

async def annotate_event_themes(
    events: List[CEKEvent],
    causal_links: List[CausalLink],
    scenes: List[Scene],
    model: str,
    client: Any
) -> None:
    """
    Main entry point for the thematic annotation stage.

    Mutates events in-place:
    - Attaches scene_id
    - Sets theme_annotations from LLM
    - Applies the Theme-Bridge Rule

    Also mutates causal_links in-place:
    - Sets edge_supertype
    """
    print(f"[theme] Attaching scene IDs to {len(events)} events...")
    attach_scene_ids_to_events(events, scenes)

    print(f"[theme] Assigning edge supertypes to {len(causal_links)} causal links...")
    assign_edge_supertypes(causal_links)

    print(f"[theme] Building local causal contexts...")
    causes_by_target: Dict[str, List[CausalLink]] = defaultdict(list)
    effects_by_source: Dict[str, List[CausalLink]] = defaultdict(list)
    for lnk in causal_links:
        causes_by_target[lnk.target_event_id].append(lnk)
        effects_by_source[lnk.source_event_id].append(lnk)
    event_map: Dict[str, CEKEvent] = {e.id: e for e in events}
    contexts = [
        build_local_causal_context(ev, causes_by_target, effects_by_source, event_map)
        for ev in events
    ]
    context_jsons = [json.dumps(ctx) for ctx in contexts]

    print(f"[theme] Annotating {len(events)} events with LLM (theme annotations)...")
    _sem = asyncio.Semaphore(20)

    async def _annotate_with_limit(ctx_json: str) -> Any:
        async with _sem:
            return await annotate_single_event_theme(ctx_json, model, client)

    tasks = [_annotate_with_limit(ctx_json) for ctx_json in context_jsons]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Attach results to events
    for event, result in zip(events, results):
        if isinstance(result, Exception):
            print(f"[theme] Warning: annotation failed for {event.id}: {result}")
            continue
        if isinstance(result, dict):
            annotations = result.get("theme_annotations", {})
            # Validate and sanitise
            clean: Dict[str, Any] = {}
            for theme in THEME_SET:
                td = annotations.get(theme, {})
                if not isinstance(td, dict):
                    td = {}
                involvement = td.get("involvement", "none")
                if involvement not in INVOLVEMENT_SET:
                    involvement = "none"
                role = td.get("role")
                if involvement == "none":
                    role = None
                elif role not in ROLE_SET:
                    role = None
                confidence = td.get("confidence")
                try:
                    confidence = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    confidence = None
                clean[theme] = {
                    "involvement": involvement,
                    "role": role,
                    "evidence": td.get("evidence", ""),
                    "signals": td.get("signals", ""),
                    "confidence": confidence,
                }
            event.theme_annotations = clean

    print(f"[theme] Applying Theme-Bridge Rule...")
    apply_theme_bridge_rule(events, causal_links)
    print(f"[theme] Theme annotation complete.")


