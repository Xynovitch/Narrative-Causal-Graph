"""
Thematic Layer v2 — annotates events with structural literary theme participation.

Themes: POWER, WEALTH, KINSHIP, JUSTICE, KNOWLEDGE
Roles: initiating, enabling, constraining, mediating, escalating, resolving, revealing
Involvement: direct, indirect, latent, none
"""
import asyncio
import json
import os
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
    # Causal production — one event directly enables or produces the next
    "CAUSES": "CAUSAL_PRODUCTION",
    "DIRECT_CAUSE": "CAUSAL_PRODUCTION",
    "ENABLES": "CAUSAL_PRODUCTION",
    "FACILITATES": "CAUSAL_PRODUCTION",
    "TRIGGERS": "CAUSAL_PRODUCTION",
    "INCITING_CAUSE": "CAUSAL_PRODUCTION",
    "EVENT_ENABLES_NEXT": "CAUSAL_PRODUCTION",
    "EVENT_REINFORCEMENT": "CAUSAL_PRODUCTION",
    "DESIRE_ALIGNMENT": "CAUSAL_PRODUCTION",
    "NECESSITATES": "CAUSAL_PRODUCTION",
    "FULFILLS": "CAUSAL_PRODUCTION",
    "PRECEDES": "CAUSAL_PRODUCTION",
    # Causal constraint — one event blocks, opposes, or limits another
    "PREVENTS": "CAUSAL_CONSTRAINT",
    "BLOCKS": "CAUSAL_CONSTRAINT",
    "INHIBITS": "CAUSAL_CONSTRAINT",
    "COMPLICATES": "CAUSAL_CONSTRAINT",
    "OPPOSES": "CAUSAL_CONSTRAINT",
    "DESIRE_OBSTRUCTION": "CAUSAL_CONSTRAINT",
    "DESIRE_COMPETITION": "CAUSAL_CONSTRAINT",
    "PHYSICAL_BLOCKAGE": "CAUSAL_CONSTRAINT",
    "INTERRUPTION_OBSTACLE": "CAUSAL_CONSTRAINT",
    "MISSION_FAILURE": "CAUSAL_CONSTRAINT",
    "MISSION_ABANDONMENT": "CAUSAL_CONSTRAINT",
    # Emotional drive — emotion or psychological state causes action
    "COMPASSION_TRIGGER": "EMOTIONAL_DRIVE",
    "EMOTIONAL_MANIPULATION": "EMOTIONAL_DRIVE",
    "EMOTIONAL_DEPENDENCE": "EMOTIONAL_DRIVE",
    "EMOTIONAL_TRIGGER": "EMOTIONAL_DRIVE",
    "EMOTIONAL_CONTAGION": "EMOTIONAL_DRIVE",
    "EMOTIONAL_DESPAIR": "EMOTIONAL_DRIVE",
    "EMOTIONAL_SUPPORT": "EMOTIONAL_DRIVE",
    "EMOTIONAL_APOLOGY": "EMOTIONAL_DRIVE",
    "EMOTIONAL_CONFESSION": "EMOTIONAL_DRIVE",
    "EMOTIONAL_ENDURANCE": "EMOTIONAL_DRIVE",
    "PSYCHOLOGICAL_IMPACT": "EMOTIONAL_DRIVE",
    "PROTECTIVE_INSTINCT": "EMOTIONAL_DRIVE",
    "CRUELTY_PLEASURE": "EMOTIONAL_DRIVE",
    "NOSTALGIA_INDUCEMENT": "EMOTIONAL_DRIVE",
    "ENRAGES": "EMOTIONAL_DRIVE",
    # Social bond — relationships, alliances, and obligations between agents
    "ALLY_DEPENDENCE": "SOCIAL_BOND",
    "ALLY_SUPPORT": "SOCIAL_BOND",
    "FAMILY_INFLUENCE": "SOCIAL_BOND",
    "FAMILY_BACKGROUND_REACTION": "SOCIAL_BOND",
    "INHERITED_OBLIGATION": "SOCIAL_BOND",
    "MENTORSHIP_SUPPORT": "SOCIAL_BOND",
    "MOTIVATES": "SOCIAL_BOND",
    "PERSUASION_ATTEMPT": "SOCIAL_BOND",
    # Narrative escalation — tension, reversal, or moral conflict raises stakes
    "CAUSES_REVERSAL": "NARRATIVE_ESCALATION",
    "ACTION_ESCALATION": "NARRATIVE_ESCALATION",
    "CONSCIENCE_CONFLICT": "NARRATIVE_ESCALATION",
    "IDENTITY_CONFLICT": "NARRATIVE_ESCALATION",
    "CONFLICT_OF_INTEREST": "NARRATIVE_ESCALATION",
    "PHYSICAL_CONFRONTATION": "NARRATIVE_ESCALATION",
    "ESCALATES": "NARRATIVE_ESCALATION",
    "COMPLICATES_FURTHER": "NARRATIVE_ESCALATION",
    "CHALLENGES": "NARRATIVE_ESCALATION",
    "MORAL_CHALLENGE": "NARRATIVE_ESCALATION",
    "MISSED_OPPORTUNITY": "NARRATIVE_ESCALATION",
    "EXPECTATION_DISAPPOINTMENT": "NARRATIVE_ESCALATION",
    "PERSONAL_TRANSFORMATION": "NARRATIVE_ESCALATION",
    "PERCEPTION_SHIFT": "NARRATIVE_ESCALATION",
    # Narrative resolution — tension closes, is explained, or redeems
    "RESOLVES": "NARRATIVE_RESOLUTION",
    "CONCLUDES": "NARRATIVE_RESOLUTION",
    "REDEEMS": "NARRATIVE_RESOLUTION",
    "PERSONAL_JOURNEY": "NARRATIVE_RESOLUTION",
    "MENTAL_RELIEF": "NARRATIVE_RESOLUTION",
    # Revelation / epistemic — information is exposed or concealed
    "REVEALS": "REVELATION_EPISTEMIC",
    "EXPOSES": "REVELATION_EPISTEMIC",
    "CONCEALS": "REVELATION_EPISTEMIC",
    "FORESHADOWS": "REVELATION_EPISTEMIC",
    "PAST_CONNECTION": "REVELATION_EPISTEMIC",
    "LOVE_INSIGHT": "REVELATION_EPISTEMIC",
    "HISTORICAL_COMPARISON": "REVELATION_EPISTEMIC",
    # Mediation / transfer — resources, knowledge, or obligation passed between agents
    "INFORMS": "MEDIATION_TRANSFER",
    "MEDIATES": "MEDIATION_TRANSFER",
    "TRANSFERS": "MEDIATION_TRANSFER",
    "DELEGATES": "MEDIATION_TRANSFER",
    "FINANCIAL_NEED": "MEDIATION_TRANSFER",
    "CULTURAL_EDUCATION": "MEDIATION_TRANSFER",
    "DECISION_MAKING": "MEDIATION_TRANSFER",
    # Thematic contrast / explanation — structural/thematic relation
    "CONTRASTS": "THEMATIC_CONTRAST",
    "MIRRORS": "THEMATIC_CONTRAST",
    "EXPLAINS": "THEMATIC_EXPLANATION",
    "SUPPORTS": "THEMATIC_EXPLANATION",
    "NARRATIVE_COMPOSITE": "THEMATIC_EXPLANATION",
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

    Records the bridge source so a reader can explain *why* the event was upgraded
    (the previous version left evidence empty, which the 0326 feedback flagged).
    """
    event_map: Dict[str, CEKEvent] = {e.id: e for e in events}

    # Build adjacency keyed by (event_id, neighbour_id) so we can recover the
    # relation_type that justifies the bridge.
    edges_out: Dict[str, List[CausalLink]] = {}
    edges_in: Dict[str, List[CausalLink]] = {}
    for lnk in causal_links:
        edges_out.setdefault(lnk.source_event_id, []).append(lnk)
        edges_in.setdefault(lnk.target_event_id, []).append(lnk)

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

            bridge_source: Optional[str] = None
            bridge_relation: Optional[str] = None
            bridge_confidence: float = 0.0

            # Outgoing: this event causes a neighbour with direct involvement.
            for lnk in edges_out.get(event.id, []):
                nb = event_map.get(lnk.target_event_id)
                if nb is None:
                    continue
                nb_td = nb.theme_annotations.get(theme, {})
                if isinstance(nb_td, dict) and nb_td.get("involvement") == "direct":
                    bridge_source = nb.id
                    bridge_relation = lnk.relation_type
                    bridge_confidence = float(nb_td.get("confidence") or 0.0)
                    break

            # Incoming: a neighbour with direct involvement causes this event.
            if bridge_source is None:
                for lnk in edges_in.get(event.id, []):
                    nb = event_map.get(lnk.source_event_id)
                    if nb is None:
                        continue
                    nb_td = nb.theme_annotations.get(theme, {})
                    if isinstance(nb_td, dict) and nb_td.get("involvement") == "direct":
                        bridge_source = nb.id
                        bridge_relation = lnk.relation_type
                        bridge_confidence = float(nb_td.get("confidence") or 0.0)
                        break

            if bridge_source is None:
                continue

            # Carry roughly half the neighbour's confidence forward, capped at 0.6.
            propagated = min(0.6, max(0.2, bridge_confidence * 0.6))

            theme_data["involvement"] = "indirect"
            theme_data["role"] = "mediating"
            theme_data["evidence"] = (
                f"Bridge: linked to {bridge_source} via {bridge_relation or 'CAUSES'} "
                f"which has direct {theme} involvement."
            )
            theme_data["signals"] = [f"bridge_from:{bridge_source}", f"via:{bridge_relation or 'CAUSES'}"]
            theme_data["confidence"] = propagated
            theme_data["bridge_source"] = bridge_source
            theme_data["bridge_relation"] = bridge_relation


# ---------------------------------------------------------------------------
# annotate_event_themes  (main entry point)
# ---------------------------------------------------------------------------

def _clean_theme_annotations(annotations_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize/validate raw LLM output into the canonical theme_annotations dict."""
    clean: Dict[str, Any] = {}
    for theme in THEME_SET:
        td = annotations_raw.get(theme, {})
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

        raw_conf = td.get("confidence")
        try:
            confidence = float(raw_conf) if raw_conf is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        if involvement == "none":
            confidence = 0.0

        raw_signals = td.get("signals", [])
        if isinstance(raw_signals, str):
            raw_signals = [raw_signals] if raw_signals else []
        elif not isinstance(raw_signals, list):
            raw_signals = []
        signals = [str(s) for s in raw_signals if s]

        evidence = td.get("evidence", "") or ""

        # Drop low-confidence direct/indirect tags (< 0.4) to "latent" — this
        # quiets the over-tagging that the feedback flagged on weak verbs.
        if involvement in {"direct", "indirect"} and confidence < 0.4:
            involvement = "latent"

        clean[theme] = {
            "involvement": involvement,
            "role": role,
            "evidence": evidence,
            "signals": signals,
            "confidence": confidence,
        }
    return clean


async def annotate_event_themes(
    events: List[CEKEvent],
    causal_links: List[CausalLink],
    scenes: List[Scene],
    model: str,
    client: Any,
    partial_checkpoint_path: Optional[str] = None,
    concurrency: int = 20,
    save_every: int = 200,
) -> None:
    """
    Main entry point for the thematic annotation stage.

    Mutates events in-place:
    - Attaches scene_id
    - Sets theme_annotations from LLM
    - Applies the Theme-Bridge Rule

    Also mutates causal_links in-place:
    - Sets edge_supertype

    If partial_checkpoint_path is given, this function will:
    - Load any prior partial state from that path on entry (resume).
    - Skip events whose annotations are already populated.
    - Periodically flush the partial state every `save_every` successes,
      so an interrupted run can be resumed without losing work.
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

    # Resume from partial checkpoint if present.
    if partial_checkpoint_path and os.path.exists(partial_checkpoint_path):
        try:
            with open(partial_checkpoint_path) as f:
                prior = json.load(f).get("theme_annotations_by_event", {})
            for ev in events:
                if ev.id in prior and prior[ev.id]:
                    ev.theme_annotations = prior[ev.id]
            print(f"[theme] Resumed partial state: {sum(1 for e in events if e.theme_annotations)} "
                  f"events already annotated, {sum(1 for e in events if not e.theme_annotations)} remaining")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[theme] Warning: could not load partial checkpoint ({e}); starting fresh")

    async def _flush_partial():
        if not partial_checkpoint_path:
            return
        payload = {
            "theme_annotations_by_event": {
                ev.id: ev.theme_annotations
                for ev in events
                if ev.theme_annotations
            }
        }
        tmp = partial_checkpoint_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, partial_checkpoint_path)
        except OSError as e:
            print(f"[theme] Warning: failed to flush partial checkpoint: {e}")

    async def _annotate_pass(items, pass_concurrency, timeout_s, label):
        if not items:
            return
        sem = asyncio.Semaphore(pass_concurrency)
        completed = 0
        failed = 0
        errors_logged = 0
        last_save_at = 0
        save_lock = asyncio.Lock()
        total = len(items)

        async def _one(ev, ctx_json):
            nonlocal completed, failed, errors_logged, last_save_at
            async with sem:
                try:
                    result = await asyncio.wait_for(
                        annotate_single_event_theme(ctx_json, model, client),
                        timeout=timeout_s,
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    failed += 1
                    if errors_logged < 5:
                        errors_logged += 1
                        print(f"[theme:{label}] error #{errors_logged} on {ev.id}: "
                              f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                    return
                if not isinstance(result, dict):
                    failed += 1
                    return
                ev.theme_annotations = _clean_theme_annotations(result.get("theme_annotations", {}))
                completed += 1
                async with save_lock:
                    if completed - last_save_at >= save_every:
                        last_save_at = completed
                        await _flush_partial()

        async def _heartbeat():
            while True:
                await asyncio.sleep(30)
                done = completed + failed
                print(f"[theme:{label}] progress: {done}/{total} "
                      f"({completed} ok, {failed} failed)", flush=True)

        print(f"[theme:{label}] Annotating {total} events "
              f"(concurrency={pass_concurrency}, timeout={timeout_s}s)...")
        hb = asyncio.create_task(_heartbeat())
        try:
            await asyncio.gather(*[_one(ev, ctx) for ev, ctx in items], return_exceptions=True)
        finally:
            hb.cancel()
            try:
                await hb
            except (asyncio.CancelledError, Exception):
                pass
            await _flush_partial()
        print(f"[theme:{label}] Done: {completed} ok, {failed} failed", flush=True)

    pending = [(ev, ctx) for ev, ctx in zip(events, context_jsons) if not ev.theme_annotations]

    if not pending:
        print(f"[theme] All {len(events)} events already annotated; skipping LLM stage")
    else:
        # Pass 1 — main run at the configured concurrency.
        await _annotate_pass(pending, concurrency, timeout_s=120, label="main")

        # Pass 2 — retry whatever still failed, more gently and with a longer timeout.
        # Most failures here are AsyncOpenAI httpx hiccups or single-call timeouts that
        # clear up on retry; this pass converts the long tail of transient errors into
        # actual annotations rather than leaving them as "none".
        ctx_by_id = {ev.id: ctx for ev, ctx in zip(events, context_jsons)}
        retry_items = [(ev, ctx_by_id[ev.id]) for ev in events if not ev.theme_annotations]
        if retry_items:
            print(f"[theme] {len(retry_items)} events failed pass 1; running retry pass.")
            await _annotate_pass(retry_items, pass_concurrency=5, timeout_s=240, label="retry")

    # Pass 3 — seed canonical "none" structure for any event still empty.
    # Without this they would be skipped by the Theme-Bridge Rule
    # (which requires an existing dict to upgrade involvement on).
    unrecoverable = [ev for ev in events if not ev.theme_annotations]
    if unrecoverable:
        print(f"[theme] Seeding default 'none' annotations for "
              f"{len(unrecoverable)} unrecoverable events so the Bridge Rule can run on them.")
        default = _clean_theme_annotations({})
        for ev in unrecoverable:
            # Fresh dict copy per event so later mutations are independent.
            ev.theme_annotations = {t: dict(default[t]) for t in THEME_SET}
        await _flush_partial()

    print(f"[theme] Applying Theme-Bridge Rule...")
    apply_theme_bridge_rule(events, causal_links)
    print(f"[theme] Theme annotation complete.")


