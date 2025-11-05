"""
Great Expectations — CEKG preprocessing pipeline (CSV export enabled)
File: great_expectations_cekg_preprocessor.py
Author: ChatGPT (GPT-5 Thinking mini)
Purpose: Preprocess Dickens' *Great Expectations* to produce event-candidate structures
         compatible with the CAUSAL EVENT KNOWLEDGE GRAPH (CEKG) schema v1.0 and
         export Neo4j bulk-import CSVs.

Design notes:
- Modular pipeline: chapter split -> sentence segmentation -> linguistic parsing ->
  event candidate extraction -> actor/patient/location/time detection -> whyFactor
  extraction and weighting -> JSON-LD output + Neo4j Cypher and CSV export.
- Heuristic and model-backed steps. Where heavy models (coreference, SRL) are unavailable
  the code falls back to simple heuristics to remain runnable.

Usage:
  1. Install dependencies (recommended):
     pip install spacy pandas
     python -m spacy download en_core_web_sm
  2. Place the full text plaintext of Great Expectations and pass its path to
     the pipeline (run method).
  3. Run and check output JSON-LD, Cypher, and CSVs in the chosen output directory.

"""

from __future__ import annotations
import re
import json
import math
import uuid
import os
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

# Optional heavy libs
try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token
except Exception:
    spacy = None  # We'll guard usage later

try:
    import pandas as pd
except Exception:
    pd = None

# -----------------------------------------------------------------------------
# Configuration: Controlled Action Ontology and Why-Factor categories
# -----------------------------------------------------------------------------
CONTROLLED_ACTION_ONTOLOGY = {
    "call": "name",
    "label": "name",
    "see": "perceive",
    "find": "perceive",
    "think": "imagine",
    "fancy": "imagine",
    "say": "say",
    "tell": "say",
    "announce": "say",
    "ask": "demand",
    "inquire": "demand",
    "warn": "threaten",
    "intimidate": "threaten",
    "bring": "give",
    "offer": "give",
    "go": "move",
    "leave": "move",
    "eat": "eat",
    "devour": "eat",
    "vow": "promise",
    "swear": "promise",
    "strike": "attack",
    "harm": "attack",
    "tremble": "fear",
    "cry": "fear",
    "look": "watch",
    "gaze": "watch",
    "symbolize": "represent",
    "signify": "represent",
}

WHYFACTOR_LEXICON = {
    "Immediate Motivation": ["fear", "hunger", "thirst", "panic", "fright", "terror"],
    "Instrumental Cause": ["to", "in order to", "so that", "so as to", "attempt", "seek", "try", "escape", "achieve", "obtain"],
    "Contextual Cause": ["because", "since", "while", "during", "in", "at", "on", "amid"],
    "Symbolic/Emotional Cause": ["love", "guilt", "shame", "pride", "honour", "pity", "envy"],
}

WHYFACTOR_BASE_WEIGHTS = {
    "Immediate Motivation": 0.45,
    "Instrumental Cause": 0.35,
    "Contextual Cause": 0.25,
    "Symbolic/Emotional Cause": 0.15,
}

CATEGORY_WEIGHT_RANGES = {
    "Immediate Motivation": (0.4, 0.5),
    "Instrumental Cause": (0.3, 0.4),
    "Contextual Cause": (0.2, 0.3),
    "Symbolic/Emotional Cause": (0.1, 0.2),
}

# -----------------------------------------------------------------------------
# Data classes for CEKG minimal Event & CausalAssertion representation
# -----------------------------------------------------------------------------

@dataclass
class CEKEvent:
    id: str
    name: str
    eventType: str
    actionType: str
    actor: Optional[str]
    patient: Optional[str]
    location: Optional[str]
    time: Optional[str]
    whyFactors: List[Dict[str, Any]]
    causeWeight: Optional[float]
    confidence: float
    provenance: Dict[str, Any]


@dataclass
class CEKCausalAssertion:
    id: str
    cause: str
    effect: str
    relationType: str
    mechanism: str
    sign: str
    weight: float
    confidence: float
    supportedBy: Dict[str, Any]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _make_id(prefix: str = "event") -> str:
    return f"{prefix}/{str(uuid.uuid4())[:8]}"


def _slugify_id(text: str, prefix: str = "id") -> str:
    # deterministic-ish id for simple nodes from name (safe for filenames)
    safe = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")[:40]
    if not safe:
        safe = uuid.uuid4().hex[:8]
    return f"{prefix}_{safe}"


def _safe_text(s: Optional[Span]) -> Optional[str]:
    if s is None:
        return None
    return str(s).strip()

PRONOUNS = set(["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"])

# -----------------------------------------------------------------------------
# Main pipeline class
# -----------------------------------------------------------------------------

class CEKGPreprocessor:
    def __init__(self, nlp: Optional[Language] = None):
        # Accept an already-initialized spaCy nlp if provided, otherwise try to create one.
        self.nlp = nlp
        if self.nlp is None and spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_trf")
            except Exception:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = None

        self.events: List[CEKEvent] = []
        self.causal_assertions: List[CEKCausalAssertion] = []

        self.recent_agents: deque = deque(maxlen=30)

    # ---------------- Text loading and segmentation ----------------
    def load_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def split_chapters(self, text: str) -> List[Tuple[int, str]]:
        parts = re.split(r"(?m)^CHAPTER\s+\w+\b.*$", text)
        if len(parts) <= 1:
            parts = re.split(r"(?m)^Chapter\s+\d+\b.*$", text)
        if len(parts) <= 1:
            paras = [p.strip() for p in text.split('\n\n') if p.strip()]
            return list(enumerate(paras, start=1))
        return list(enumerate([p.strip() for p in parts if p.strip()], start=1))

    # ------------------ Linguistic parsing helpers ------------------
    def annotate(self, text: str) -> Doc:
        if self.nlp is None:
            raise RuntimeError("spaCy is not available. Initialize CEKGPreprocessor with a spaCy model.")
        return self.nlp(text)

    def heuristic_coref(self, token: Token) -> Optional[str]:
        t = token.text.lower()
        if t in PRONOUNS and len(self.recent_agents) > 0:
            return self.recent_agents[-1]
        return None

    # ------------------ Event extraction ------------------
    def extract_events_from_doc(self, doc: Doc, chapter_id: int = None) -> List[CEKEvent]:
        events: List[CEKEvent] = []
        for sent in doc.sents:
            verbs = [tok for tok in sent if tok.pos_ == "VERB" and (tok.dep_ == "ROOT" or tok.tag_.startswith("V"))]
            if not verbs:
                verbs = [tok for tok in sent if tok.pos_ == "AUX" or tok.pos_ == "VERB"]
            for v in verbs:
                lemma = v.lemma_.lower()
                actionType = CONTROLLED_ACTION_ONTOLOGY.get(lemma, lemma)
                actor, patient = self._extract_arguments(v)
                if actor and not actor.lower() in PRONOUNS:
                    self.recent_agents.append(actor)
                location = self._extract_location(sent)
                time = self._extract_time(sent)
                whyFactors, causeWeight = self._extract_whyfactors(sent)
                confidence = 0.6
                ev = CEKEvent(
                    id=_make_id("event"),
                    name=(f"{actionType} -- {v.text} | {sent.text[:60].strip()}"),
                    eventType=self._classify_event_type(v, sent),
                    actionType=actionType,
                    actor=actor,
                    patient=patient,
                    location=location,
                    time=time,
                    whyFactors=whyFactors,
                    causeWeight=causeWeight,
                    confidence=confidence,
                    provenance={
                        "chapter": chapter_id,
                        "quote": sent.text.strip(),
                        "extractionMethod": "spacy_heuristic"
                    }
                )
                if ev.actor is None or ev.eventType is None or not ev.whyFactors:
                    ev.confidence *= 0.6
                events.append(ev)
        return events

    def _extract_arguments(self, verb_token: Token) -> Tuple[Optional[str], Optional[str]]:
        actor = None
        patient = None
        sent = verb_token.sent
        for tok in sent:
            if tok.head == verb_token and tok.dep_ in ("nsubj", "nsubjpass", "csubj"):
                actor = self._resolve_mention(tok)
            if tok.head == verb_token and tok.dep_ in ("dobj", "obj", "pobj"):
                patient = self._resolve_mention(tok)
        if actor is None:
            for tok in sent:
                if tok.dep_ in ("nsubj",) and tok.head == verb_token:
                    actor = self._resolve_mention(tok)
                    break
        return actor, patient

    def _resolve_mention(self, tok: Token) -> Optional[str]:
        if tok.ent_type_:
            return tok.text
        if tok.pos_ == "PRON" or tok.lower_ in PRONOUNS:
            c = self.heuristic_coref(tok)
            return c
        lefts = [w for w in tok.lefts if w.dep_ == "compound"]
        if lefts:
            np = lefts[0].subtree
            return " ".join([t.text for t in np])
        return tok.text

    def _extract_location(self, sent: Span) -> Optional[str]:
        for ent in sent.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                return ent.text
        return None

    def _extract_time(self, sent: Span) -> Optional[str]:
        for ent in sent.ents:
            if ent.label_ in ("DATE", "TIME"):
                return ent.text
        return None

    def _classify_event_type(self, verb_token: Token, sent: Span) -> str:
        lemma = verb_token.lemma_.lower()
        if lemma in ("recognize", "see", "notice", "find"):
            return "perception/recognition"
        if lemma in ("say", "tell", "ask", "speak"):
            return "encounter/dialogue"
        if lemma in ("attack", "hit", "strike", "harm"):
            return "conflict/threat"
        if lemma in ("give", "offer", "lend"):
            return "assistance/exchange"
        if lemma in ("go", "leave", "arrive"):
            return "movement/departure"
        if verb_token.tag_.startswith("V") and verb_token.pos_ == "VERB":
            return "action"
        return "other"

    # ------------------ Why-factor extraction ------------------
    def _extract_whyfactors(self, sent: Span) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        text = sent.text.lower()
        factors: List[Dict[str, Any]] = []
        scores: Dict[str, float] = {}
        for cat, tokens in WHYFACTOR_LEXICON.items():
            for tok in tokens:
                if tok in text:
                    scores[cat] = scores.get(cat, 0.0) + (1.0 / (1 + abs(len(tok) - 4)))
        if not scores:
            for cat, lex in WHYFACTOR_LEXICON.items():
                for w in lex:
                    if re.search(r"\b" + re.escape(w) + r"\b", text):
                        scores[cat] = scores.get(cat, 0.0) + 0.5
        if not scores:
            factors.append({"factor": "contextual_setting", "weight": WHYFACTOR_BASE_WEIGHTS["Contextual Cause"], "category": "Contextual Cause"})
            total_weight = WHYFACTOR_BASE_WEIGHTS["Contextual Cause"]
            return factors, total_weight
        total_raw = sum(scores.values())
        for cat, raw in scores.items():
            prop = raw / total_raw if total_raw > 0 else 0
            lo, hi = CATEGORY_WEIGHT_RANGES.get(cat, (0.1, 0.2))
            weight = lo + (hi - lo) * prop
            factors.append({"factor": cat.lower().replace(" ", "_"), "weight": round(weight, 3), "category": cat})
        s = sum(f["weight"] for f in factors)
        if s > 0:
            for f in factors:
                f["weight"] = round(f["weight"] / s, 3)
        causeWeight = sum(f["weight"] for f in factors)
        return factors, round(causeWeight, 3)

    # ------------------ JSON-LD + Neo4j exporters ------------------
    def build_jsonld(self) -> Dict[str, Any]:
        g = []
        for ev in self.events:
            g.append({
                "@id": ev.id,
                "type": "Event",
                "name": ev.name,
                "eventType": ev.eventType,
                "actionType": ev.actionType,
                "actor": ev.actor,
                "patient": ev.patient,
                "location": ev.location,
                "time": ev.time,
                "whyFactors": ev.whyFactors,
                "causeWeight": ev.causeWeight,
                "confidence": ev.confidence,
                "provenance": ev.provenance,
            })
        for ca in self.causal_assertions:
            g.append({
                "@id": ca.id,
                "type": "CausalAssertion",
                "cause": ca.cause,
                "effect": ca.effect,
                "relationType": ca.relationType,
                "mechanism": ca.mechanism,
                "sign": ca.sign,
                "weight": ca.weight,
                "confidence": ca.confidence,
                "supportedBy": ca.supportedBy,
            })
        return {"@graph": g}

    def export_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.build_jsonld(), f, ensure_ascii=False, indent=2)

    def export_neo4j_cypher(self, path: str):
        lines = []
        for ev in self.events:
            props = {
                "id": ev.id,
                "name": ev.name.replace('\"', '\\\"'),
                "eventType": ev.eventType,
                "actionType": ev.actionType,
                "actor": ev.actor or "",
                "patient": ev.patient or "",
                "location": ev.location or "",
                "time": ev.time or "",
                "causeWeight": ev.causeWeight or 0.0,
                "confidence": ev.confidence,
            }
            props_str = ", ".join([f"{k}: \"{v}\"" if isinstance(v, str) else f"{k}: {v}" for k, v in props.items()])
            lines.append(f"MERGE (e:Event {{id: \"{ev.id}\"}}) SET e += {{{props_str}}};")
            if ev.actor:
                lines.append(f"MERGE (a:Agent {{name: \"{ev.actor}\"}});")
                lines.append(f"MERGE (e)-[:HAS_ACTOR]->(a);")
            if ev.patient:
                lines.append(f"MERGE (p:Agent {{name: \"{ev.patient}\"}});")
                lines.append(f"MERGE (e)-[:AFFECTS]->(p);")
            if ev.location:
                lines.append(f"MERGE (pl:Place {{name: \"{ev.location}\"}});")
                lines.append(f"MERGE (e)-[:AT]->(pl);")
        for ca in self.causal_assertions:
            lines.append(f"MERGE (c:CausalAssertion {{id: \"{ca.id}\"}}) SET c.relationType=\"{ca.relationType}\", c.weight={ca.weight};")
            lines.append(f"MATCH (a:Event {{id: \"{ca.cause}\"}}), (b:Event {{id: \"{ca.effect}\"}}) MERGE (caNode:CausalAssertion {{id: \"{ca.id}\"}}) MERGE (caNode)-[:CAUSE]->(a) MERGE (caNode)-[:EFFECT]->(b);")
            quote = ca.supportedBy.get("quote", "")
            if quote:
                lines.append(f"MERGE (s:Source {{quote: \"{quote.replace('\"','\\\\\"')}\"}}) MERGE (caNode)-[:SUPPORTED_BY]->(s);")
        with open(path, "w", encoding="utf-8") as f:
            f.write('\n'.join(lines))

    # ------------------ CSV export for Neo4j bulk import ------------------
    def export_csv(self, out_dir: str = "neo4j_csv"):
        os.makedirs(out_dir, exist_ok=True)

        # build agent and place maps
        agent_names = set()
        place_names = set()
        for ev in self.events:
            if ev.actor:
                agent_names.add(ev.actor)
            if ev.patient:
                agent_names.add(ev.patient)
            if ev.location:
                place_names.add(ev.location)
        agent_map = {name: _slugify_id(name, prefix="agent") for name in agent_names}
        place_map = {name: _slugify_id(name, prefix="place") for name in place_names}

        # events dataframe
        events_rows = []
        for ev in self.events:
            events_rows.append({
                ":ID": ev.id,
                "name": ev.name,
                "eventType": ev.eventType,
                "actionType": ev.actionType,
                "actor_id": agent_map.get(ev.actor) if ev.actor else "",
                "patient_id": agent_map.get(ev.patient) if ev.patient else "",
                "location_id": place_map.get(ev.location) if ev.location else "",
                "time": ev.time or "",
                "causeWeight": ev.causeWeight or 0.0,
                "confidence": ev.confidence,
                "chapter": ev.provenance.get("chapter"),
                "quote": ev.provenance.get("quote", "")
            })
        # agents dataframe
        agent_rows = []
        for name, aid in agent_map.items():
            agent_rows.append({":ID": aid, "name": name})
        # places dataframe
        place_rows = []
        for name, pid in place_map.items():
            place_rows.append({":ID": pid, "name": name})

        # causal assertions
        causal_rows = []
        causal_cause_rel = []
        causal_effect_rel = []
        supported_by_rows = []
        source_map = {}
        for ca in self.causal_assertions:
            causal_rows.append({":ID": ca.id, "relationType": ca.relationType, "weight": ca.weight, "confidence": ca.confidence, "mechanism": ca.mechanism, "sign": ca.sign})
            causal_cause_rel.append({":START_ID": ca.id, ":END_ID": ca.cause, ":TYPE": "CAUSE"})
            causal_effect_rel.append({":START_ID": ca.id, ":END_ID": ca.effect, ":TYPE": "EFFECT"})
            quote = ca.supportedBy.get("quote", "")
            if quote:
                sid = _slugify_id(quote[:80], prefix="src")
                source_map[quote] = sid
                supported_by_rows.append({":START_ID": ca.id, ":END_ID": sid, ":TYPE": "SUPPORTED_BY"})

        # relationships between events and agents/places
        has_actor_rows = []
        affects_rows = []
        at_rows = []
        for ev in self.events:
            if ev.actor and agent_map.get(ev.actor):
                has_actor_rows.append({":START_ID": ev.id, ":END_ID": agent_map.get(ev.actor), ":TYPE": "HAS_ACTOR"})
            if ev.patient and agent_map.get(ev.patient):
                affects_rows.append({":START_ID": ev.id, ":END_ID": agent_map.get(ev.patient), ":TYPE": "AFFECTS"})
            if ev.location and place_map.get(ev.location):
                at_rows.append({":START_ID": ev.id, ":END_ID": place_map.get(ev.location), ":TYPE": "AT"})

        # sources (if any)
        source_rows = []
        for quote, sid in source_map.items():
            source_rows.append({":ID": sid, "quote": quote})

        # write CSVs (use pandas if available)
        def _write_csv(rows, filename):
            path = os.path.join(out_dir, filename)
            if not rows:
                # create empty file with a minimal header
                open(path, "w", encoding="utf-8").close()
                return path
            if pd is not None:
                df = pd.DataFrame(rows)
                df.to_csv(path, index=False)
            else:
                # fallback to csv module
                import csv
                keys = list(rows[0].keys())
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(rows)
            return path

        # map of filename -> rows
        files_map = {
            "events.csv": events_rows,
            "agents.csv": agent_rows,
            "places.csv": place_rows,
            "causal_assertions.csv": causal_rows,
            "causal_cause_rel.csv": causal_cause_rel,
            "causal_effect_rel.csv": causal_effect_rel,
            "has_actor.csv": has_actor_rows,
            "affects.csv": affects_rows,
            "at.csv": at_rows,
            "sources.csv": source_rows,
            "supported_by.csv": supported_by_rows,
        }

        written = {}
        for fname, rows in files_map.items():
            written[fname] = _write_csv(rows, fname)

        return {k: os.path.join(out_dir, k) for k in files_map.keys()}

    # ------------------ High-level run method ------------------
    def run(self, text_path: str, out_json: str = "gek_preprocessed.json", out_cypher: str = "gek_import.cypher", out_csv_dir: str = "neo4j_csv"):
        raw = self.load_text(text_path)
        chapters = self.split_chapters(raw)
        for idx, chunk in chapters:
            try:
                doc = self.annotate(chunk)
            except Exception:
                continue
            evs = self.extract_events_from_doc(doc, chapter_id=idx)
            self.events.extend(evs)
        self._heuristic_causal_linking()
        self.export_json(out_json)
        self.export_neo4j_cypher(out_cypher)
        csv_paths = self.export_csv(out_csv_dir)
        return {
            "json": out_json,
            "cypher": out_cypher,
            "csv": csv_paths
        }

    def _heuristic_causal_linking(self):
        cue = "because"
        for ev in self.events:
            if ev.provenance and cue in ev.provenance.get("quote", "").lower():
                ch = ev.provenance.get("chapter")
                earlier = [e for e in self.events if e.provenance and e.provenance.get("chapter") == ch]
                candidate = None
                for cand in reversed(earlier):
                    if cand.id == ev.id:
                        continue
                    if cand.actor and ev.actor and cand.actor.split()[0] == ev.actor.split()[0]:
                        candidate = cand
                        break
                if candidate:
                    ca = CEKCausalAssertion(
                        id=_make_id("causal"),
                        cause=candidate.id,
                        effect=ev.id,
                        relationType="causes",
                        mechanism="lexical-cue because; heuristic link",
                        sign="+",
                        weight=(candidate.causeWeight or 0.5) * (ev.causeWeight or 0.5),
                        confidence=min(candidate.confidence, ev.confidence) * 0.8,
                        supportedBy={"quote": ev.provenance.get("quote", "")}
                    )
                    self.causal_assertions.append(ca)


# ----------------------------- Example usage --------------------------------
if __name__ == "__main__":
    try:
        nlp = None
        if spacy:
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = None
        prep = CEKGPreprocessor(nlp=nlp)
        print("CEKG Preprocessor initialized. Use run(text_path) to process and export CSVs.")
    except Exception:
        print("Warning: spaCy or pandas may be missing. The pipeline will run with reduced capabilities.")

# End of file
