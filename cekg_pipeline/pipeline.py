import asyncio
import traceback
import os
import json
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple

from . import config, schemas, utils, text_processor, llm_service, graph_builder, graph_mapper, exporters
from .coreference_resolver import get_resolver

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
        self.coref_resolver = get_resolver()
        
        # Load custom ontologies
        self.event_ontology = self._load_ontology("event_ontology.json", "event_types")
        self.relationship_ontology = self._load_ontology("relationship_ontology.json", "relationship_types")
        if self.event_ontology:
            print(f"[config] Loaded custom Event Ontology ({len(self.event_ontology)} types).")
        if self.relationship_ontology:
            print(f"[config] Loaded custom Relationship Ontology ({len(self.relationship_ontology)} types).")

    def _load_ontology(self, filename: str, key: str) -> Optional[List[str]]:
        """Load custom ontology from JSON file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f).get(key)
            except Exception as e:
                print(f"[warning] Failed to load {filename}: {e}")
        return None
    
    def _calculate_calibrated_confidence(self, event_data, logprobs):
        """Calculate calibrated confidence using multiple signals"""
        p_llm = float(event_data.get("confidence", 0.7))
        
        # Lexical score based on completeness
        score = 1.0
        if not event_data.get("actors"): score -= 0.2
        if not event_data.get("location_context"): score -= 0.1
        p_lexical = score
        
        # Contextual score using semantic similarity
        p_contextual = 0.5
        if SENTENCE_TRANSFORMER_MODEL:
            try:
                desc_emb = SENTENCE_TRANSFORMER_MODEL.encode(
                    event_data.get("raw_description", ""), convert_to_tensor=True
                )
                quote_emb = SENTENCE_TRANSFORMER_MODEL.encode(
                    event_data.get("source_quote", ""), convert_to_tensor=True
                )
                p_contextual = max(0, util.pytorch_cos_sim(desc_emb, quote_emb).item())
            except:
                pass
        
        return round((0.4 * p_llm) + (0.4 * p_lexical) + (0.2 * p_contextual), 4)

    def _parse_event_json_data(self, event_data_list, chapter_id, logprobs, 
                               enable_confidence_calibration, graph_model="star"):
        """Parse event JSON data into schema objects with coreference resolution"""
        all_events = []
        all_produces = []
        entity_occurrences_batch = defaultdict(list)
        
        if not isinstance(event_data_list, list):
            event_data_list = []
        
        logprobs_list = [logprobs] * len(event_data_list) if logprobs else [None] * len(event_data_list)
        
        # First pass: collect all actor mentions for learning
        all_actor_mentions = []
        for event_data in event_data_list:
            actors = event_data.get("actors", [])
            if actors:
                all_actor_mentions.append([str(a) for a in actors if isinstance(a, str)])
        
        # Learn from co-occurrence patterns
        if all_actor_mentions:
            self.coref_resolver.learn_from_cooccurrence(all_actor_mentions)
        
        # Second pass: process events with resolution
        for i, event_data in enumerate(event_data_list):
            try:
                # Calculate confidence
                confidence = self._calculate_calibrated_confidence(
                    event_data, logprobs_list[i]
                ) if enable_confidence_calibration else float(event_data.get("confidence", 0.7))
                
                # Process why_factors
                raw_why = event_data.get("why_factors", [])
                why_factors_list = [str(w) for w in raw_why] if isinstance(raw_why, list) else []
                
                seq = self.global_event_sequence
                self.global_event_sequence += 1
                
                # Create event
                event = schemas.CEKEvent(
                    id=utils._make_id("event"),
                    raw_description=event_data.get("raw_description", event_data.get("name", "Untitled")),
                    event_category=event_data.get("event_category", "OTHER"),
                    action_type=event_data.get("event_category", "OTHER"),
                    time_context=event_data.get("time_context"),
                    location_context=event_data.get("location_context"),
                    actors=[], patients=[], why_factors=why_factors_list,
                    chapter=chapter_id, sequence=seq, confidence=confidence,
                    source_quote=event_data.get("quote", "")
                )
                
                # Process actors with coreference resolution
                raw_actors = event_data.get("actors", [])
                if isinstance(raw_actors, str):
                    raw_actors = [raw_actors]
                
                resolved_actors = self.coref_resolver.batch_resolve(
                    [str(a) for a in raw_actors if isinstance(a, str)]
                )
                
                clean_actors = []
                for actor_name in resolved_actors:
                    clean_actors.append(actor_name)
                    aid = graph_builder._generate_entity_id(actor_name, "agent", event.id, graph_model)
                    all_produces.append(schemas.EventProducesEntity(
                        event.id, aid, actor_name, "actor", "PRODUCES_ACTOR", 1.0
                    ))
                    entity_occurrences_batch[f"actor:{actor_name.lower()}"].append((event.id, seq))
                
                event.actors = clean_actors

                # Process patients with coreference resolution
                raw_patients = event_data.get("patients", [])
                if isinstance(raw_patients, str):
                    raw_patients = [raw_patients]
                
                resolved_patients = self.coref_resolver.batch_resolve(
                    [str(p) for p in raw_patients if isinstance(p, str)]
                )
                
                clean_patients = []
                for pat_name in resolved_patients:
                    clean_patients.append(pat_name)
                    pid = graph_builder._generate_entity_id(pat_name, "agent", event.id, graph_model)
                    all_produces.append(schemas.EventProducesEntity(
                        event.id, pid, pat_name, "patient", "PRODUCES_PATIENT", 1.0
                    ))
                    entity_occurrences_batch[f"patient:{pat_name.lower()}"].append((event.id, seq))
                
                event.patients = clean_patients
                
                # Process why_factors (no resolution needed)
                for wf in why_factors_list:
                    wid = graph_builder._generate_entity_id(wf[:30], "why", event.id, graph_model)
                    all_produces.append(schemas.EventProducesEntity(
                        event.id, wid, wf, "whyfactor", "PRODUCES_MOTIVATION", 1.0
                    ))
                    entity_occurrences_batch[f"whyfactor:{wf.lower()}"].append((event.id, seq))

                all_events.append(event)
            except Exception as e:
                print(f"[warning] Event parse error: {e}")
                traceback.print_exc()
                continue
        
        print(f"[coref] Resolved {len(all_events)} events. Resolver stats: {self.coref_resolver.get_statistics()}")
        
        return all_events, all_produces, entity_occurrences_batch

    async def _process_text_chunk(self, text_chunk, chapter_id, enable_llm_expansion, 
                                  enable_confidence_calibration, extraction_style, graph_model):
        """Process a single text chunk"""
        try:
            data, logprobs = await llm_service.extract_events_from_text(
                text_chunk, chapter_id, self.openai_model, self.client,
                enable_llm_expansion, False, extraction_style, self.event_ontology
            )
            return self._parse_event_json_data(
                data, chapter_id, logprobs, enable_confidence_calibration, graph_model
            )
        except Exception as e:
            print(f"[error] Chunk processing failed: {e}")
            return [], [], defaultdict(list)

    async def _process_parallel_batch(self, paragraphs_with_chapter, enable_llm_expansion,
                                      enable_confidence_calibration, extraction_style, graph_model):
        """Process multiple paragraphs in parallel"""
        all_events, all_produces, entity_occurrences = [], [], defaultdict(list)
        try:
            results = await llm_service.batch_extract_events(
                paragraphs_with_chapter, self.openai_model, self.client,
                enable_llm_expansion, False, extraction_style, self.event_ontology
            )
            for (json_data, logprobs), (para, cid) in zip(results, paragraphs_with_chapter):
                evs, prods, occs = self._parse_event_json_data(
                    json_data, cid, logprobs, enable_confidence_calibration, graph_model
                )
                all_events.extend(evs)
                all_produces.extend(prods)
                for k, v in occs.items():
                    entity_occurrences[k].extend(v)
        except Exception as e:
            print(f"[error] Batch processing failed: {e}")
        return all_events, all_produces, entity_occurrences

    async def _batch_causal_linking(self, events, entity_occurrences, window, 
                                    sample_rate, batch_size):
        """Perform causal linking between events"""
        self.dag_validator.add_events(events)
        pairs_set = set()
        
        # Window-based pairs
        for i, ev in enumerate(events):
            start = max(0, i - window)
            for j in range(start, i):
                if events[j].sequence < ev.sequence:
                    pairs_set.add((events[j].id, ev.id))
        
        # Entity co-occurrence pairs
        for k, occs in entity_occurrences.items():
            if k.startswith("actor:") or k.startswith("patient:"):
                for i in range(len(occs) - 1):
                    c_id, c_seq = occs[i]
                    e_id, e_seq = occs[i + 1]
                    if c_seq < e_seq:
                        pairs_set.add((c_id, e_id))
        
        print(f"[causal linking] Found {len(pairs_set)} candidate pairs.")
        
        ev_map = {e.id: e for e in events}
        pairs_list = [(ev_map[c].raw_description, ev_map[e].raw_description, c, e) 
                     for c, e in pairs_set if c in ev_map and e in ev_map]
        
        causal_links = []
        for i in range(0, len(pairs_list), batch_size):
            batch = pairs_list[i:i + batch_size]
            results = await llm_service.batch_assess_pairs(
                batch, self.openai_model, self.client, self.relationship_ontology
            )
            
            for (c_txt, e_txt, c_id, e_id), res in zip(batch, results):
                raw_rt = res.get("relationType") if res else None
                if isinstance(raw_rt, list):
                    raw_rt = raw_rt[0] if raw_rt else "NONE"
                
                if res and raw_rt not in [None, "NONE"]:
                    if self.dag_validator.add_edge(c_id, e_id):
                        rt_str = str(raw_rt).upper()
                        causal_links.append(schemas.CausalLink(
                            c_id, e_id, rt_str, res.get("mechanism", ""),
                            float(res.get("weight", 0)), float(res.get("confidence", 0))
                        ))
            print(f"[causal] Processed {min(i + batch_size, len(pairs_list))}/{len(pairs_list)} pairs")
        
        return causal_links

    async def _batch_semantic_linking(self, events, batch_size):
        """Perform semantic (non-causal) linking between events"""
        pairs = []
        for i, ev in enumerate(events):
            for j in range(max(0, i - 4), i):
                pairs.append((events[j].raw_description, ev.raw_description, events[j].id, ev.id))
        
        links = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            results = await llm_service.batch_assess_semantic_pairs(batch, self.openai_model, self.client)
            
            for (c, e, cid, eid), res in zip(batch, results):
                if not res:
                    continue
                
                raw_rel = res.get("relation")
                if isinstance(raw_rel, list):
                    raw_rel = raw_rel[0] if raw_rel else "none"
                
                if raw_rel and str(raw_rel).lower() != "none":
                    links.append(schemas.SemanticLink(
                        utils._make_id("sem"), [cid], [eid], str(raw_rel),
                        res.get("cue"), float(res.get("confidence", 0))
                    ))
        return links

    async def _generate_scenes(self, events):
        """Group events into narrative scenes with aggregated participant info"""
        scenes = []
        by_chap = defaultdict(list)
        for e in events:
            by_chap[e.chapter].append(e)
        
        for cid, evs in by_chap.items():
            data = await llm_service.extract_scenes_from_chapter_async(
                evs, cid, self.openai_model, self.client
            )
            for d in data:
                try:
                    event_ids = d.get("event_ids", [])
                    
                    # Aggregate participants from included events
                    participants = set()
                    locations = set()
                    times = set()
                    
                    for ev in evs:
                        if ev.id in event_ids:
                            participants.update(ev.actors)
                            participants.update(ev.patients)
                            if ev.location_context:
                                locations.add(ev.location_context)
                            if ev.time_context:
                                times.add(ev.time_context)
                    
                    # Choose primary location/time (most common or first)
                    primary_location = list(locations)[0] if locations else None
                    time_period = list(times)[0] if times else None
                    
                    scenes.append(schemas.Scene(
                        utils._make_id("scene"), cid, event_ids,
                        primary_location, time_period, list(participants),
                        d.get("theme", ""), "",
                        float(d.get("confidence", 0))
                    ))
                except Exception as e:
                    print(f"[warning] Failed to create scene: {e}")
        
        print(f"[scenes] Generated {len(scenes)} scenes with aggregated participant info")
        return scenes

    async def run_async(self, text_path, out_json, out_cypher, out_csv_dir,
                       max_chapters, batch_size, causal_window, causal_sample_rate,
                       causal_batch_size, paragraph_chunk_size, extraction_style,
                       graph_model, enable_scene_grouping, enable_semantic_linking,
                       enable_llm_expansion, enable_confidence_calibration):
        """Main pipeline execution"""
        print(f"[pipeline] Loading text... (Graph Model: {graph_model})")
        raw = text_processor.load_text(text_path)
        chapters = text_processor.split_chapters(raw)[:max_chapters] if max_chapters else text_processor.split_chapters(raw)
        
        all_events, all_produces, entity_occurrences = [], [], defaultdict(list)
        self.global_event_sequence = 0

        for cid, txt in chapters:
            paras = text_processor.split_into_paragraphs(txt)
            print(f"[chapter {cid}] Processing {len(paras)} paragraphs...")
            
            if paragraph_chunk_size != 1:
                csize = len(paras) if paragraph_chunk_size == 0 else paragraph_chunk_size
                for i in range(0, len(paras), csize):
                    chunk = "\n\n".join(paras[i:i + csize])
                    e, p, o = await self._process_text_chunk(
                        chunk, cid, enable_llm_expansion, enable_confidence_calibration,
                        extraction_style, graph_model
                    )
                    all_events.extend(e)
                    all_produces.extend(p)
                    for k, v in o.items():
                        entity_occurrences[k].extend(v)
            else:
                for i in range(0, len(paras), batch_size):
                    batch = [(p, cid) for p in paras[i:i + batch_size]]
                    e, p, o = await self._process_parallel_batch(
                        batch, enable_llm_expansion, enable_confidence_calibration,
                        extraction_style, graph_model
                    )
                    all_events.extend(e)
                    all_produces.extend(p)
                    for k, v in o.items():
                        entity_occurrences[k].extend(v)

        print(f"[pipeline] Total events: {len(all_events)}")
        
        # Propagate context
        all_events = graph_builder.propagate_context_attributes(all_events)
        new_prods, entity_occurrences = graph_builder.propagate_context(
            all_events, all_produces, entity_occurrences, graph_model=graph_model
        )
        all_produces.extend(new_prods)

        # Create entity-to-event links
        e2e_links = graph_builder.create_entity_to_event_links(
            entity_occurrences, all_produces, graph_model=graph_model
        )
        
        # Causal linking
        causal_links = await self._batch_causal_linking(
            all_events, entity_occurrences, causal_window, causal_sample_rate, causal_batch_size
        )
        
        # Optional features
        scenes = await self._generate_scenes(all_events) if enable_scene_grouping else []
        sem_links = await self._batch_semantic_linking(all_events, causal_batch_size) if enable_semantic_linking else []

        # Export
        print(f"[pipeline] Mapping to Generic Graph...")
        generic_nodes, generic_rels = graph_mapper.map_to_generic_graph(
            all_events, all_produces, e2e_links, causal_links,
            graph_model=graph_model, semantic_links=sem_links, scenes=scenes
        )

        print(f"[pipeline] Exporting results...")
        exporters.export_json(out_json, exporters.build_jsonld(all_events, all_produces, e2e_links, causal_links))
        exporters.export_neo4j_cypher(out_cypher, generic_nodes, generic_rels)
        exporters.export_csv(out_csv_dir, all_events, all_produces, e2e_links, causal_links, 
                            sem_links, scenes, graph_model=graph_model)

        return {"stats": {"events": len(all_events), "characters": len(self.coref_resolver.character_registry)}}