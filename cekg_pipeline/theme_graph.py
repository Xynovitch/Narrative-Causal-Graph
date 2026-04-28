"""
Thematic Edge Layer
===================

Themes are *not* separate nodes; they are properties of edges that connect
Event nodes into thematic subplots.

The intuition:
- Each Event already carries a `theme_annotations` dict on the node
  (POWER / WEALTH / KINSHIP / JUSTICE / KNOWLEDGE → involvement, role, ...).
- A *subplot* is a chain of Event nodes bound by a shared theme. Edges
  along that chain are the subplot beats.
- A thematic edge is therefore an Event → Event link with the theme stored
  as a property on the edge itself, alongside the source/target roles and
  involvement.

Two complementary edge sources are combined:

1. **Causal-projected thematic edges.** For each existing causal link
   (cause → effect), if both endpoints have non-none involvement for some
   theme T, we emit a thematic edge with theme=T. The causal beat *is* the
   subplot beat — the edge stores how the cause-side and effect-side roles
   carry the theme forward.

2. **Sequential thematic spine within scenes.** Inside each scene, sort the
   events by sequence and, for each theme T, link consecutive theme-active
   events. This captures subplot beats that aren't separately marked as
   causal but still belong to the theme's spine within a scene.

Both modes write to a single relationship type, `THEMATIC_LINK`, with
`theme` as a property. Cypher queries:

    // all WEALTH subplot beats:
    MATCH (a:Event)-[r:THEMATIC_LINK {theme: 'WEALTH'}]->(b:Event)
    RETURN a, r, b

    // POWER beats where the cause initiates and the effect escalates:
    MATCH (a)-[r:THEMATIC_LINK {theme: 'POWER',
                                 source_role: 'initiating',
                                 target_role: 'escalating'}]->(b)
    RETURN a, r, b
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import schemas

THEMES: Tuple[str, ...] = ("POWER", "WEALTH", "KINSHIP", "JUSTICE", "KNOWLEDGE")

_ACTIVE = ("direct", "indirect")


def _theme_data(ev: schemas.CEKEvent, theme: str) -> Dict[str, Any]:
    ann = ev.theme_annotations or {}
    if not isinstance(ann, dict):
        return {}
    td = ann.get(theme, {})
    return td if isinstance(td, dict) else {}


def _is_theme_active(ev: schemas.CEKEvent, theme: str) -> bool:
    td = _theme_data(ev, theme)
    return (td.get("involvement") or "none").lower() in _ACTIVE


def _safe_conf(td: Dict[str, Any]) -> float:
    try:
        return float(td.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_thematic_event_edges(
    events: List[schemas.CEKEvent],
    causal_links: Optional[List[schemas.CausalLink]] = None,
    scenes: Optional[List[schemas.Scene]] = None,
    enable_sequential_spine: bool = True,
) -> List[schemas.GenericRelationship]:
    """
    Build Event → Event thematic edges. Theme is a property on the edge.

    The connected component induced by edges sharing the same `theme` value
    is a subplot.

    Edge property contract:
        theme               — one of POWER / WEALTH / KINSHIP / JUSTICE / KNOWLEDGE
        source_role         — role of theme on the source event ("" if none)
        target_role         — role of theme on the target event ("" if none)
        source_involvement  — direct | indirect
        target_involvement  — direct | indirect
        source_confidence   — float
        target_confidence   — float
        confidence          — sqrt(source*target) — combined signal
        via                 — provenance: "causal:<RELATION_TYPE>" or
                              "scene_spine:<scene_id>"
        scene_id            — scene of the source event (if any)
        sequence_distance   — |target.sequence - source.sequence|
    """
    edges: List[schemas.GenericRelationship] = []
    seen: set = set()

    event_map: Dict[str, schemas.CEKEvent] = {e.id: e for e in events}

    def _emit(src: schemas.CEKEvent, tgt: schemas.CEKEvent, theme: str, via: str) -> None:
        key = (src.id, tgt.id, theme)
        if key in seen:
            return
        seen.add(key)

        src_td = _theme_data(src, theme)
        tgt_td = _theme_data(tgt, theme)
        sc = _safe_conf(src_td)
        tc = _safe_conf(tgt_td)
        combined = (sc * tc) ** 0.5 if sc and tc else max(sc, tc) * 0.5

        edges.append(schemas.GenericRelationship(
            start_node_uid=src.id,
            end_node_uid=tgt.id,
            rel_type="THEMATIC_LINK",
            properties={
                "theme": theme,
                "source_role": (src_td.get("role") or "") or "",
                "target_role": (tgt_td.get("role") or "") or "",
                "source_involvement": (src_td.get("involvement") or "none").lower(),
                "target_involvement": (tgt_td.get("involvement") or "none").lower(),
                "source_confidence": round(sc, 4),
                "target_confidence": round(tc, 4),
                "confidence": round(combined, 4),
                "via": via,
                "scene_id": src.scene_id or "",
                "sequence_distance": abs((tgt.sequence or 0) - (src.sequence or 0)),
            },
        ))

    # ---- Mode 1: causal-projected thematic edges ----
    if causal_links:
        for link in causal_links:
            src = event_map.get(link.source_event_id)
            tgt = event_map.get(link.target_event_id)
            if src is None or tgt is None:
                continue
            for theme in THEMES:
                if _is_theme_active(src, theme) and _is_theme_active(tgt, theme):
                    _emit(src, tgt, theme, via=f"causal:{link.relation_type}")

    # ---- Mode 2: sequential thematic spine within scenes ----
    if enable_sequential_spine and scenes:
        for scene in scenes:
            scene_events = [
                event_map[eid]
                for eid in scene.included_event_ids
                if eid in event_map
            ]
            scene_events.sort(key=lambda e: (e.chapter or 0, e.sequence or 0))
            for theme in THEMES:
                spine = [e for e in scene_events if _is_theme_active(e, theme)]
                for i in range(len(spine) - 1):
                    _emit(spine[i], spine[i + 1], theme, via=f"scene_spine:{scene.id}")

    return edges
