import asyncio
import traceback
import random
import os
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple

from . import config
from . import schemas
from . import utils
from . import text_processor
from . import llm_service
from . import graph_builder
from . import graph_mapper
from . import exporters

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    SENTENCE_TRANSFORMER_MODEL = None
    print("[warning] 'sentence-transformers' library not found.")

class CEKGPreprocessor:
    
    def __init__(self, openai_model: Optional[str] = None):
        if not config.OPENAI_API_KEY:
             raise RuntimeError("OPENAI_API_KEY not set.")
        
        self.openai_model = openai_model or config.OPENAI_MODEL
        self.client = llm_service.init_openai_client(config.OPENAI_API_KEY)
        self.dag_validator = utils.DAGValidator()
        self.global_event_sequence = 0
        
        # --- LOAD DYNAMIC ONTOLOGIES ---
        # Matches the output format of generate_ontology.py and generate_relationship_ontology.py
        self.event_ontology = self._load_ontology("event_ontology.json", "event_types")
        self.relationship_ontology = self._load_ontology("relationship_ontology.json", "relationship_types")
        
        if self.event_ontology:
            print(f"[config] Loaded custom Event Ontology ({len(self.event_ontology)} types).")
        if self.relationship_ontology:
            print(f"[config] Loaded custom Relationship Ontology ({len(self.relationship_ontology)} types).")

    def _load_ontology(self, filename: str, key: str) -> Optional[List[str]]:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get(key)
            except Exception as e:
                print(f"[warning] Failed to load {filename}: {e}")
        return None
    
    def _calculate_calibrated_confidence(self, event_data: Dict[str, Any], logprobs: Optional[Any]) -> float:
        p_llm = float(event_data.get("confidence", 0.7))
        score = 1.0
        if not event_data.get("actors"): score -= 0.2
        if not event_data.get("location_context"): score -= 0.1
        p_lexical = score
        p_contextual = 0.5
        if SENTENCE_TRANSFORMER_MODEL:
            try:
                n = SENTENCE_TRANSFORMER_MODEL.encode(event_data.get("raw_description", ""), convert_to_tensor=True)
                q = SENTENCE_TRANSFORMER_MODEL.encode(event_data.get("source_quote", ""), convert_to_tensor=True)
                p_contextual = max(0, util.pytorch_cos_sim(n, q).item())
            except: pass
        return round((0.4 * p_llm) + (0.4 * p_lexical) + (0.2 * p_contextual), 4)

    def _parse_event_json_data(self, event_data_list, chapter_id, logprobs, enable_confidence_calibration):
        all_events = []
        all_produces = []
        entity_occurrences_batch = defaultdict(list)
        if not isinstance(event_data_list, list): event_data_list = []
        logprobs_list = [logprobs] * len(event_data_list) if logprobs else [None] * len(event_data_list)
        
        for i, event_data in enumerate(event_data_list):
            try:
                confidence = self._calculate_calibrated_confidence(event_data, logprobs_list[i]) if enable_confidence_calibration else float(event_data.get("confidence", 0.7))
                
                raw_why = event_data.get("why_factors", [])
                why_factors_list = [str(w) for w in raw_why] if isinstance(raw_why, list) else []
                loc_ctx = event_data.get("location_context")
                time_ctx = event_data.get("time_context")
                seq = self.global_event_sequence
                self.global_event_sequence += 1
                
                event = schemas.CEKEvent(
                    id=utils._make_id("event"),
                    raw_description=event_data.get("raw_description", event_data.get("name", "Untitled")),
                    event_category=event_data.get("event_category", "OTHER"),
                    action_type=event_data.get("event_category", "OTHER"),
                    time_context=time_ctx, location_context=loc_ctx,
                    actors=[], patients=[], why_factors=why_factors_list,
                    chapter=chapter_id, sequence=seq, confidence=confidence,
                    source_quote=event_data.get("quote", "")
                )
                
                # Entities
                clean_actors = []
                for actor_name in event_data.get("actors", []):
                    if isinstance(actor_name, str) and actor_name.strip():
                        name = actor_name.strip()
                        clean_actors.append(name)
                        aid = utils._make_id(f"agent_{name.lower().replace(' ', '_')}")
                        all_produces.append(schemas.EventProducesEntity(event.id, aid, name, "actor", "PRODUCES_ACTOR", 1.0))
                        entity_occurrences_batch[f"actor:{name.lower()}"].append((event.id, seq))
                event.actors = clean_actors

                clean_patients = []
                for pat_name in event_data.get("patients", []):
                    if isinstance(pat_name, str) and pat_name.strip():
                        name = pat_name.strip()
                        clean_patients.append(name)
                        pid = utils._make_id(f"agent_{name.lower().replace(' ', '_')}")
                        all_produces.append(schemas.EventProducesEntity(event.id, pid, name, "patient", "PRODUCES_PATIENT", 1.0))
                        entity_occurrences_batch[f"patient:{name.lower()}"].append((event.id, seq))
                event.patients = clean_patients
                
                for wf in why_factors_list:
                    wid = utils._make_id(f"why_{wf.lower()[:30].replace(' ', '_')}")
                    all_produces.append(schemas.EventProducesEntity(event.id, wid, wf, "whyfactor", "PRODUCES_MOTIVATION", 1.0))
                    entity_occurrences_batch[f"whyfactor:{wf.lower()}"].append((event.id, seq))

                all_events.append(event)
            except Exception as e:
                print(f"[warning] Event parse error: {e}")
                continue
        return all_events, all_produces, entity_occurrences_batch

    async def _process_parallel_batch(self, paragraphs_with_chapter, enable_llm_expansion, enable_confidence_calibration, extraction_style):
        all_events, all_produces = [], []
        entity_occurrences = defaultdict(list)
        try:
            results = await llm_service.batch_extract_events(
                paragraphs_with_chapter, self.openai_model, self.client, 
                enable_llm_expansion, False, extraction_style,
                event_ontology=self.event_ontology # <--- Pass Ontology
            )
            for (json_data, logprobs), (para, cid) in zip(results, paragraphs_with_chapter):
                evs, prods, occs = self._parse_event_json_data(json_data, cid, logprobs, enable_confidence_calibration)
                all_events.extend(evs); all_produces.extend(prods)
                for k,v in occs.items(): entity_occurrences[k].extend(v)
        except Exception as e: print(f"[error] Batch processing failed: {e}")
        return all_events, all_produces, entity_occurrences
    
    async def _process_text_chunk(self, text_chunk, chapter_id, enable_llm_expansion, enable_confidence_calibration, extraction_style):
        try:
            data, logprobs = await llm_service.extract_events_from_text(
                text_chunk, chapter_id, self.openai_model, self.client, 
                enable_llm_expansion, False, extraction_style,
                event_ontology=self.event_ontology # <--- Pass Ontology
            )
            return self._parse_event_json_data(data, chapter_id, logprobs, enable_confidence_calibration)
        except Exception as e:
            print(f"[error] Chunk processing failed: {e}")
            return [], [], defaultdict(list)

    async def _batch_causal_linking(self, events, entity_occurrences, window, sample_rate, batch_size):
        self.dag_validator.add_events(events)
        pairs_set = set()
        
        # Pass 1 & 2
        for i, ev in enumerate(events):
            start = max(0, i - window)
            for j in range(start, i):
                if events[j].sequence < ev.sequence: pairs_set.add((events[j].id, ev.id))
        
        for k, occs in entity_occurrences.items():
            if k.startswith("actor:") or k.startswith("patient:"):
                for i in range(len(occs)-1):
                    c_id, c_seq = occs[i]; e_id, e_seq = occs[i+1]
                    if c_seq < e_seq: pairs_set.add((c_id, e_id))
        
        print(f"[causal linking] Found {len(pairs_set)} candidate pairs.")
        ev_map = {e.id: e for e in events}
        pairs_list = []
        for c_id, e_id in pairs_set:
            if c_id in ev_map and e_id in ev_map:
                pairs_list.append((ev_map[c_id].raw_description, ev_map[e_id].raw_description, c_id, e_id))
        
        causal_links = []
        for i in range(0, len(pairs_list), batch_size):
            batch = pairs_list[i:i+batch_size]
            results = await llm_service.batch_assess_pairs(
                batch, self.openai_model, self.client,
                relationship_ontology=self.relationship_ontology # <--- Pass Ontology
            )
            
            for (c_txt, e_txt, c_id, e_id), res in zip(batch, results):
                if res and res.get("relationType") not in [None, "NONE"]:
                    if self.dag_validator.add_edge(c_id, e_id):
                        rt_str = res.get("relationType", "DIRECT_CAUSE").upper()
                        causal_links.append(schemas.CausalLink(
                            c_id, e_id, rt_str, res.get("mechanism",""), float(res.get("weight",0)), float(res.get("confidence",0))
                        ))
            print(f"[causal] Processed {min(i+batch_size, len(pairs_list))}/{len(pairs_list)} pairs")
        return causal_links

    async def _batch_semantic_linking(self, events, batch_size):
        pairs = []
        for i, ev in enumerate(events):
            for j in range(max(0, i-4), i):
                pairs.append((events[j].raw_description, ev.raw_description, events[j].id, ev.id))
        links = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            results = await llm_service.batch_assess_semantic_pairs(batch, self.openai_model, self.client)
            for (c,e,cid,eid), res in zip(batch, results):
                if res and res.get("relation") != "none":
                    links.append(schemas.SemanticLink(utils._make_id("sem"), [cid], [eid], res.get("relation"), res.get("cue"), float(res.get("confidence",0))))
        return links

    async def _generate_scenes(self, events):
        scenes = []
        by_chap = defaultdict(list)
        for e in events: by_chap[e.chapter].append(e)
        for cid, evs in by_chap.items():
            data = await llm_service.extract_scenes_from_chapter_async(evs, cid, self.openai_model, self.client)
            for d in data:
                try: scenes.append(schemas.Scene(utils._make_id("scene"), cid, d.get("event_ids",[]), None, None, [], d.get("theme",""), "", float(d.get("confidence",0))))
                except: pass
        return scenes

    async def run_async(self, text_path, out_json, out_cypher, out_csv_dir, max_chapters, batch_size, causal_window, causal_sample_rate, causal_batch_size, paragraph_chunk_size, extraction_style, graph_model, enable_scene_grouping, enable_semantic_linking, enable_llm_expansion, enable_confidence_calibration):
        print("[pipeline] Loading text...")
        raw = text_processor.load_text(text_path)
        chapters = text_processor.split_chapters(raw)[:max_chapters] if max_chapters else text_processor.split_chapters(raw)
        
        all_events, all_produces = [], []
        entity_occurrences = defaultdict(list)
        self.global_event_sequence = 0

        # --- Extraction Phase ---
        for cid, txt in chapters:
            paras = text_processor.split_into_paragraphs(txt)
            print(f"[chapter {cid}] Processing {len(paras)} paragraphs...")
            if paragraph_chunk_size != 1:
                csize = len(paras) if paragraph_chunk_size == 0 else paragraph_chunk_size
                for i in range(0, len(paras), csize):
                    chunk = "\n\n".join(paras[i:i+csize])
                    try:
                        e, p, o = await self._process_text_chunk(chunk, cid, enable_llm_expansion, enable_confidence_calibration, extraction_style)
                        all_events.extend(e); all_produces.extend(p)
                        for k,v in o.items(): entity_occurrences[k].extend(v)
                        print(f"[chapter {cid}] Chunk {i//csize + 1}: extracted {len(e)} events")
                    except Exception as err: print(f"[error] Chunk failed: {err}")
            else:
                for i in range(0, len(paras), batch_size):
                    batch = [(p, cid) for p in paras[i:i+batch_size]]
                    try:
                        e, p, o = await self._process_parallel_batch(batch, enable_llm_expansion, enable_confidence_calibration, extraction_style)
                        all_events.extend(e); all_produces.extend(p)
                        for k,v in o.items(): entity_occurrences[k].extend(v)
                        print(f"[chapter {cid}] Batch {i//batch_size + 1}: extracted {len(e)} events")
                    except Exception as err: print(f"[error] Batch failed: {err}")

        print(f"[pipeline] Total events: {len(all_events)}")
        
        # --- Context Propagation ---
        print("[pipeline] Propagating context attributes...")
        all_events = graph_builder.propagate_context_attributes(all_events)
        new_prods, entity_occurrences = graph_builder.propagate_context(all_events, all_produces, entity_occurrences)
        all_produces.extend(new_prods)

        # --- Linking ---
        print("[pipeline] Creating links...")
        e2e_links = graph_builder.create_entity_to_event_links(entity_occurrences, all_produces)
        causal_links = await self._batch_causal_linking(all_events, entity_occurrences, causal_window, causal_sample_rate, causal_batch_size)
        
        scenes = await self._generate_scenes(all_events) if enable_scene_grouping else []
        sem_links = await self._batch_semantic_linking(all_events, causal_batch_size) if enable_semantic_linking else []

        # --- Export ---
        exporters.export_json(out_json, exporters.build_jsonld(all_events, all_produces, e2e_links, causal_links))
        exporters.export_neo4j_cypher(out_cypher, all_events, all_produces, e2e_links, causal_links, sem_links, scenes)
        exporters.export_csv(out_csv_dir, all_events, all_produces, e2e_links, causal_links, sem_links, scenes)

        return {"stats": {"events": len(all_events)}}