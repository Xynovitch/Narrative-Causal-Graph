"""
Enhanced Pipeline with Intelligent Long-Range Causal Inference
Fixed: Replaced O(N²) brute force with O(N log N) smart filtering
"""

import asyncio
import traceback
import os
import json
import random
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple

from . import config, schemas, utils, text_processor, llm_service, graph_builder, graph_mapper, exporters
from .coreference_resolver import get_resolver
from .ontology_loader import get_ontology_manager, OntologyManager

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    SENTENCE_TRANSFORMER_MODEL = None
    print("[warning] 'sentence-transformers' library not found.")

THEORY_MCKEE = "@McKee"
THEORY_TRUBY = "@Truby"
THEORY_MCKEE_LOWER = "mckee"
THEORY_TRUBY_LOWER = "truby"

def normalize_theory_name(theory: str) -> str:
    if not theory:
        return THEORY_MCKEE
    theory_lower = theory.lower().replace("@", "")
    if theory_lower == THEORY_MCKEE_LOWER:
        return THEORY_MCKEE
    elif theory_lower == THEORY_TRUBY_LOWER:
        return THEORY_TRUBY
    else:
        return THEORY_MCKEE

class CEKGPreprocessor:
    def __init__(self, openai_model: Optional[str] = None, schema_path: Optional[str] = None):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.openai_model = openai_model or config.OPENAI_MODEL
        self.client = llm_service.init_openai_client(config.OPENAI_API_KEY)
        self.dag_validator = utils.DAGValidator()
        self.global_event_sequence = 0
        self.coref_resolver = get_resolver()
        self.ontology = get_ontology_manager(schema_path)
        self.event_ontology = self.ontology.get_event_type_names()
        self.mckee_relations = self.ontology.get_relation_type_names(THEORY_MCKEE_LOWER)
        self.truby_relations = self.ontology.get_relation_type_names(THEORY_TRUBY_LOWER)
        
        print(f"[pipeline] Initialized with {len(self.event_ontology)} event types")
        print(f"[pipeline] McKee relations: {len(self.mckee_relations)}")
        print(f"[pipeline] Truby relations: {len(self.truby_relations)}")
    
    def _calculate_calibrated_confidence(self, event_data, logprobs):
        p_llm = float(event_data.get("confidence", 0.7))
        score = 1.0
        if not event_data.get("actors"): score -= 0.2
        if not event_data.get("location_context"): score -= 0.1
        p_lexical = score
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

    def _infer_theory_from_event_type(self, event_type: str) -> str:
        if event_type in self.ontology.event_types:
            return normalize_theory_name(self.ontology.event_types[event_type].theory)
        return THEORY_MCKEE

    def _parse_event_json_data(self, event_data_list, chapter_id, logprobs, 
                               enable_confidence_calibration, graph_model="star"):
        all_events = []
        all_produces = []
        entity_occurrences_batch = defaultdict(list)
        
        if not isinstance(event_data_list, list):
            if isinstance(event_data_list, dict):
                if 'events' in event_data_list:
                    event_data_list = event_data_list['events']
                else:
                    event_data_list = []
            else:
                event_data_list = []
        
        if not event_data_list:
            return all_events, all_produces, entity_occurrences_batch
        
        logprobs_list = [logprobs] * len(event_data_list) if logprobs else [None] * len(event_data_list)
        
        all_actor_mentions = []
        for event_data in event_data_list:
            actors = event_data.get("actors", [])
            if actors:
                all_actor_mentions.append([str(a) for a in actors if isinstance(a, str)])
        
        if all_actor_mentions:
            self.coref_resolver.learn_from_cooccurrence(all_actor_mentions)
        
        for i, event_data in enumerate(event_data_list):
            try:
                confidence = self._calculate_calibrated_confidence(
                    event_data, logprobs_list[i]
                ) if enable_confidence_calibration else float(event_data.get("confidence", 0.7))
                
                event_type = event_data.get("event_category", "OTHER")
                if not self.ontology.validate_event_type(event_type):
                    event_type = "PHYSICAL_ACTION"
                
                theory = self._infer_theory_from_event_type(event_type)
                raw_why = event_data.get("why_factors", [])
                why_factors_list = [str(w) for w in raw_why] if isinstance(raw_why, list) else []
                
                seq = self.global_event_sequence
                self.global_event_sequence += 1
                
                event = schemas.CEKEvent(
                    id=utils._make_id("event"),
                    raw_description=event_data.get("raw_description", event_data.get("name", "Untitled")),
                    event_category=event_type,
                    action_type=event_type,
                    time_context=event_data.get("time_context"),
                    location_context=event_data.get("location_context"),
                    actors=[], patients=[], why_factors=why_factors_list,
                    chapter=chapter_id, sequence=seq, confidence=confidence,
                    source_quote=event_data.get("quote", ""),
                    theory=theory
                )
                
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
                        event.id, aid, actor_name, "actor", "PRODUCES_ACTOR", 1.0,
                        agent_type=None, theory=theory
                    ))
                    entity_occurrences_batch[f"actor:{actor_name.lower()}"].append((event.id, seq))
                
                event.actors = clean_actors

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
                        event.id, pid, pat_name, "patient", "PRODUCES_PATIENT", 1.0,
                        agent_type=None, theory=theory
                    ))
                    entity_occurrences_batch[f"patient:{pat_name.lower()}"].append((event.id, seq))
                
                event.patients = clean_patients
                
                for wf in why_factors_list:
                    wid = graph_builder._generate_entity_id(wf[:30], "why", event.id, graph_model)
                    all_produces.append(schemas.EventProducesEntity(
                        event.id, wid, wf, "whyfactor", "PRODUCES_MOTIVATION", 1.0,
                        theory=theory
                    ))
                    entity_occurrences_batch[f"whyfactor:{wf.lower()}"].append((event.id, seq))

                all_events.append(event)
            except Exception as e:
                print(f"[warning] Event parse error: {e}")
                continue
        
        return all_events, all_produces, entity_occurrences_batch

    async def _process_text_chunk(self, text_chunk, chapter_id, enable_llm_expansion, 
                                  enable_confidence_calibration, extraction_style, graph_model):
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

    async def _classify_agent_types(self, events: List[schemas.CEKEvent], 
                                    event_produces: List[schemas.EventProducesEntity]) -> Dict[str, str]:
        print("[agent_classification] Analyzing character roles...")
        character_events = defaultdict(list)
        for event in events:
            for actor in event.actors:
                character_events[actor].append(event.raw_description)
            for patient in event.patients:
                character_events[patient].append(event.raw_description)
        
        classifications = {}
        agent_type_names = self.ontology.get_agent_type_names()
        
        if not agent_type_names:
            print("[warning] No agent types defined, skipping classification")
            return classifications
        
        for char_name, char_events in character_events.items():
            if len(char_events) < 2:
                continue
            
            sample_events = char_events[:10] if len(char_events) > 10 else char_events
            
            try:
                agent_type = await llm_service.classify_agent_type(
                    char_name, sample_events, agent_type_names,
                    self.openai_model, self.client
                )
                
                if self.ontology.validate_agent_type(agent_type):
                    classifications[char_name] = agent_type
                    print(f"[agent_classification] {char_name} → {agent_type}")
            except Exception as e:
                print(f"[warning] Failed to classify {char_name}: {e}")
        
        for prod in event_produces:
            if prod.entity_type in ['actor', 'patient']:
                if prod.entity_name in classifications:
                    prod.agent_type = classifications[prod.entity_name]
        
        return classifications

    async def _batch_causal_linking_mixed_theory(self, events, entity_occurrences, 
                                             window, sample_rate, batch_size,
                                             enable_long_range=False,
                                             theory_mode="mixed",
                                             max_concurrent_calls=10,
                                             max_pairs=50000):
        from .optimized_linking import intelligent_long_range_linking
        
        self.dag_validator.add_events(events)
        all_causal_links = []
        mckee_link_count = 0
        truby_link_count = 0
        
        if enable_long_range:
            print("[long_range] Using INTELLIGENT filtering...")
            
            theories_to_process = []
            if theory_mode == "mixed":
                theories_to_process = [
                    (THEORY_MCKEE_LOWER, self.mckee_relations, THEORY_MCKEE), 
                    (THEORY_TRUBY_LOWER, self.truby_relations, THEORY_TRUBY)
                ]
            elif theory_mode == THEORY_MCKEE_LOWER:
                theories_to_process = [(THEORY_MCKEE_LOWER, self.mckee_relations, THEORY_MCKEE)]
            elif theory_mode == THEORY_TRUBY_LOWER:
                theories_to_process = [(THEORY_TRUBY_LOWER, self.truby_relations, THEORY_TRUBY)]
            
            for theory_name, relation_ontology, theory_tag in theories_to_process:
                if not relation_ontology:
                    continue
                
                links, count = await intelligent_long_range_linking(
                    events=events,
                    assess_pairs_bulk_func=llm_service.assess_pairs_bulk,
                    model=self.openai_model,
                    client=self.client,
                    relation_ontology=relation_ontology,
                    theory_name=theory_name,
                    dag_validator=self.dag_validator,
                    ontology_validator=self.ontology,
                    entity_occurrences=entity_occurrences,
                    max_pairs=max_pairs,
                    max_concurrent_calls=max_concurrent_calls
                )
                
                all_causal_links.extend(links)
                
                if theory_tag == THEORY_MCKEE:
                    mckee_link_count += count
                elif theory_tag == THEORY_TRUBY:
                    truby_link_count += count
        else:
            # Short-range mode
            pairs_set = set()
            
            for i, ev in enumerate(events):
                start = max(0, i - window)
                for j in range(start, i):
                    if events[j].sequence < ev.sequence and events[j].chapter == ev.chapter:
                        pairs_set.add((events[j].id, ev.id))
            
            for k, occs in entity_occurrences.items():
                if k.startswith("actor:") or k.startswith("patient:"):
                    for i in range(len(occs) - 1):
                        c_id, c_seq = occs[i]
                        e_id, e_seq = occs[i + 1]
                        if c_seq < e_seq:
                            pairs_set.add((c_id, e_id))
            
            pairs_list = list(pairs_set)
            
            if len(pairs_list) > 5000:
                random.shuffle(pairs_list)
                pairs_list = pairs_list[:5000]
            
            ev_map = {e.id: e for e in events}
            pairs_with_text = [
                (ev_map[c].raw_description, ev_map[e].raw_description, c, e)
                for c, e in pairs_list if c in ev_map and e in ev_map
            ]
            
            theories_to_process = []
            if theory_mode == "mixed":
                theories_to_process = [
                    (THEORY_MCKEE_LOWER, self.mckee_relations, THEORY_MCKEE),
                    (THEORY_TRUBY_LOWER, self.truby_relations, THEORY_TRUBY)
                ]
            elif theory_mode == THEORY_MCKEE_LOWER:
                theories_to_process = [(THEORY_MCKEE_LOWER, self.mckee_relations, THEORY_MCKEE)]
            elif theory_mode == THEORY_TRUBY_LOWER:
                theories_to_process = [(THEORY_TRUBY_LOWER, self.truby_relations, THEORY_TRUBY)]
            
            BULK_SIZE = 20
            for theory_name, relation_ontology, theory_tag in theories_to_process:
                if not relation_ontology:
                    continue
                
                print(f"\n[{theory_name}] Analyzing {len(pairs_with_text)} pairs...")
                
                for i in range(0, len(pairs_with_text), BULK_SIZE):
                    batch_pairs = pairs_with_text[i:i + BULK_SIZE]
                    
                    results = await llm_service.assess_pairs_bulk(
                        batch_pairs, self.openai_model, self.client, relation_ontology
                    )
                    
                    for (c_txt, e_txt, c_id, e_id), res in zip(batch_pairs, results):
                        if not res:
                            continue
                        
                        raw_rt = res.get("relationType")
                        if raw_rt and str(raw_rt).upper() not in ["NONE", "NULL", "None"]:
                            rt_str = str(raw_rt).upper()
                            
                            if not self.ontology.validate_relation_type(rt_str, theory_name):
                                continue
                            
                            directionality = self.ontology.get_relation_directionality(rt_str, theory_name)
                            
                            if self.dag_validator.add_edge(c_id, e_id):
                                all_causal_links.append(schemas.CausalLink(
                                    c_id, e_id, rt_str, res.get("mechanism", ""),
                                    float(res.get("weight", 0)),
                                    float(res.get("confidence", 0)),
                                    theory=theory_tag,
                                    directionality=directionality
                                ))
                                
                                if theory_tag == THEORY_MCKEE:
                                    mckee_link_count += 1
                                elif theory_tag == THEORY_TRUBY:
                                    truby_link_count += 1
        
        print(f"[causal linking] Created {len(all_causal_links)} total links")
        if theory_mode == "mixed":
            print(f"  - McKee: {mckee_link_count}, Truby: {truby_link_count}")
        
        return all_causal_links, mckee_link_count, truby_link_count

    async def _batch_semantic_linking(self, events, batch_size):
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

    async def _generate_scenes_with_all_entities(self, events, event_produces):
        print("[scenes] Creating scene-centric structure...")
        scenes = []
        by_chap = defaultdict(list)
        for e in events:
            by_chap[e.chapter].append(e)
        
        event_to_entities = defaultdict(lambda: {"actors": set(), "patients": set(), "whyfactors": set()})
        for prod in event_produces:
            if prod.entity_type == "actor":
                event_to_entities[prod.event_id]["actors"].add(prod.entity_name)
            elif prod.entity_type == "patient":
                event_to_entities[prod.event_id]["patients"].add(prod.entity_name)
            elif prod.entity_type == "whyfactor":
                event_to_entities[prod.event_id]["whyfactors"].add(prod.entity_name)
        
        for cid, evs in by_chap.items():
            data = await llm_service.extract_scenes_from_chapter_async(
                evs, cid, self.openai_model, self.client
            )
            
            if not data:
                data = [{
                    "event_ids": [e.id for e in evs],
                    "theme": f"Chapter {cid} narrative",
                    "confidence": 0.5
                }]
            
            event_ids_in_chapter = {e.id for e in evs}
            
            for d in data:
                try:
                    event_ids = d.get("event_ids", [])
                    valid_event_ids = [eid for eid in event_ids if eid in event_ids_in_chapter]
                    
                    if not valid_event_ids:
                        continue
                    
                    all_actors = set()
                    all_patients = set()
                    all_whyfactors = set()
                    locations = set()
                    times = set()
                    
                    for ev in evs:
                        if ev.id in valid_event_ids:
                            entities = event_to_entities[ev.id]
                            all_actors.update(entities["actors"])
                            all_patients.update(entities["patients"])
                            all_whyfactors.update(entities["whyfactors"])
                            
                            if ev.location_context:
                                locations.add(ev.location_context)
                            if ev.time_context:
                                times.add(ev.time_context)
                    
                    primary_location = list(locations)[0] if locations else None
                    time_period = list(times)[0] if times else None
                    all_participants = list(all_actors | all_patients)
                    
                    scene = schemas.Scene(
                        utils._make_id("scene"), cid, valid_event_ids,
                        primary_location, time_period, all_participants,
                        d.get("theme", ""), "",
                        float(d.get("confidence", 0)),
                        place_type=None, time_type=None
                    )
                    
                    scene.all_actors = list(all_actors)
                    scene.all_patients = list(all_patients)
                    scene.all_whyfactors = list(all_whyfactors)
                    
                    scenes.append(scene)
                except Exception as e:
                    print(f"[warning] Failed to create scene: {e}")
        
        all_event_ids_in_scenes = set()
        for scene in scenes:
            all_event_ids_in_scenes.update(scene.included_event_ids)
        
        orphan_events = [e for e in events if e.id not in all_event_ids_in_scenes]
        if orphan_events:
            by_chapter = defaultdict(list)
            for ev in orphan_events:
                by_chapter[ev.chapter].append(ev)
            
            for cid, evs in by_chapter.items():
                event_ids = [e.id for e in evs]
                all_actors = set()
                all_patients = set()
                all_whyfactors = set()
                
                for ev in evs:
                    entities = event_to_entities[ev.id]
                    all_actors.update(entities["actors"])
                    all_patients.update(entities["patients"])
                    all_whyfactors.update(entities["whyfactors"])
                
                scene = schemas.Scene(
                    utils._make_id("scene"), cid, event_ids,
                    None, None, list(all_actors | all_patients),
                    f"Chapter {cid} miscellaneous", "",
                    0.3, None, None
                )
                scene.all_actors = list(all_actors)
                scene.all_patients = list(all_patients)
                scene.all_whyfactors = list(all_whyfactors)
                scenes.append(scene)
        
        print(f"[scenes] Generated {len(scenes)} scenes")
        return scenes

    async def run_async(self, text_path, out_json, out_cypher, out_csv_dir,
                   max_chapters=None, batch_size=5, causal_window=10, 
                   causal_sample_rate=0.5, causal_batch_size=10, 
                   paragraph_chunk_size=1, extraction_style="detailed",
                   graph_model="star",
                   enable_mixed_theory=True,
                   enable_agent_classification=False,
                   enable_long_range_inference=False,
                   enable_scene_grouping=True,
                   enable_semantic_linking=False,
                   enable_llm_expansion=True,
                   enable_confidence_calibration=True,
                   max_concurrent_calls=10,
                   max_long_range_pairs=50000):
        
        theory_mode = "mixed" if enable_mixed_theory else THEORY_MCKEE_LOWER
        mode_desc = "Mixed Theory" if enable_mixed_theory else "Single Theory"
        
        print(f"[pipeline] Loading text...")
        print(f"[pipeline] Graph Model: {graph_model}")
        print(f"[pipeline] Causal Mode: {mode_desc}")
        print(f"[pipeline] Long-Range: {'ON' if enable_long_range_inference else 'OFF'}")
        if enable_long_range_inference:
            print(f"[pipeline] Max pairs cap: {max_long_range_pairs:,}")
        
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
        
        all_events = graph_builder.propagate_context_attributes(all_events)
        new_prods, entity_occurrences = graph_builder.propagate_context(
            all_events, all_produces, entity_occurrences, graph_model=graph_model
        )
        all_produces.extend(new_prods)
        
        entity_to_event_links = graph_builder.create_entity_to_event_links(
            entity_occurrences, all_produces, graph_model=graph_model
        )
        
        agent_classifications = {}
        if enable_agent_classification:
            agent_classifications = await self._classify_agent_types(all_events, all_produces)
        
        print(f"\n[causal] Starting causal analysis...")
        causal_links, mckee_count, truby_count = await self._batch_causal_linking_mixed_theory(
            all_events, entity_occurrences, causal_window, causal_sample_rate,
            causal_batch_size, enable_long_range_inference, theory_mode, 
            max_concurrent_calls, max_long_range_pairs
        )
        
        semantic_links = []
        if enable_semantic_linking:
            semantic_links = await self._batch_semantic_linking(all_events, causal_batch_size)
        
        scenes = []
        if enable_scene_grouping:
            scenes = await self._generate_scenes_with_all_entities(all_events, all_produces)
        
        all_characters = set()

        for prod in all_produces:
            if prod.entity_type in ['actor', 'patient']:
                all_characters.add(prod.entity_name)
        
        print("\n[export] Exporting results...")
        
        jsonld = exporters.build_jsonld(
            all_events, all_produces, entity_to_event_links, causal_links
        )
        exporters.export_json(out_json, jsonld)
        
        csv_paths = exporters.export_csv(
            out_csv_dir, all_events, all_produces, entity_to_event_links,
            causal_links, semantic_links, scenes, graph_model
        )
        
        nodes, relationships = graph_mapper.map_to_generic_graph(
            all_events, all_produces, entity_to_event_links, causal_links,
            graph_model, semantic_links, scenes, agent_classifications
        )
        
        exporters.export_neo4j_cypher(out_cypher, nodes, relationships)
        
        dag_stats = self.dag_validator.get_stats()
        print(f"\n[dag] DAG Statistics:")
        print(f"  Nodes: {dag_stats['nodes']}")
        print(f"  Edges: {dag_stats['edges']}")
        print(f"  Max In-Degree: {dag_stats['max_in_degree']}")
        print(f"  Avg In-Degree: {dag_stats['avg_in_degree']}")
        
        is_valid_dag = self.dag_validator.validate_dag()
        print(f"  Is Valid DAG: {is_valid_dag}")
        
        cache_stats = await llm_service.get_cache_sizes()
        
        return {
            "stats": {
                "events": len(all_events),
                "characters": len(all_characters),
                "causal_links": len(causal_links),
                "mckee_links": mckee_count if enable_mixed_theory else len(causal_links),
                "truby_links": truby_count if enable_mixed_theory else 0,
                "semantic_links": len(semantic_links),
                "scenes": len(scenes),
                "agent_types_classified": len(agent_classifications),
                "dag_valid": is_valid_dag,
                "dag_stats": dag_stats,
                "cache_stats": cache_stats
            },
            "events": all_events,
            "causal_links": causal_links,
            "semantic_links": semantic_links,
            "scenes": scenes,
            "agent_classifications": agent_classifications,
            "csv_paths": csv_paths
        }