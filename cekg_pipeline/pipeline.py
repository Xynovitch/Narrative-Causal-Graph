"""
Enhanced Pipeline with:
1. Mixed Theory Graphs (McKee + Truby simultaneously)
2. Automatic Agent Type Classification (optional)
3. Scene-Centric Structure (all entities/events under scenes)
4. Long-Range Causal Inference (cross-chapter)

FIXES:
- Added fallback for single-theory mode
- Fixed theory capitalization consistency
- Improved error handling for missing ontologies
- Better validation for LLM responses
"""

import asyncio
import traceback
import os
import json
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

class CEKGPreprocessor:
    def __init__(self, openai_model: Optional[str] = None, schema_path: Optional[str] = None):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.openai_model = openai_model or config.OPENAI_MODEL
        self.client = llm_service.init_openai_client(config.OPENAI_API_KEY)
        self.dag_validator = utils.DAGValidator()
        self.global_event_sequence = 0
        self.coref_resolver = get_resolver()
        
        # Load schema-based ontologies
        self.ontology = get_ontology_manager(schema_path)
        
        # Use schema event types and relation types
        self.event_ontology = self.ontology.get_event_type_names()
        self.mckee_relations = self.ontology.get_relation_type_names("mckee")
        self.truby_relations = self.ontology.get_relation_type_names("truby")
        
        print(f"[pipeline] Initialized with {len(self.event_ontology)} event types")
        print(f"[pipeline] McKee relations: {len(self.mckee_relations)}")
        print(f"[pipeline] Truby relations: {len(self.truby_relations)}")
    
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

    def _infer_theory_from_event_type(self, event_type: str) -> str:
        """Infer theory attribution from event type"""
        if event_type in self.ontology.event_types:
            return self.ontology.event_types[event_type].theory
        return "@McKee"  # Default

    def _parse_event_json_data(self, event_data_list, chapter_id, logprobs, 
                               enable_confidence_calibration, graph_model="star"):
        """Parse event JSON data into schema objects with theory attribution"""
        all_events = []
        all_produces = []
        entity_occurrences_batch = defaultdict(list)
        
        # FIX: Better validation of event_data_list
        if not isinstance(event_data_list, list):
            if isinstance(event_data_list, dict):
                # Try to extract events from dict
                if 'events' in event_data_list:
                    event_data_list = event_data_list['events']
                else:
                    print(f"[warning] Unexpected dict structure, keys: {event_data_list.keys()}")
                    event_data_list = []
            else:
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
        
        # Second pass: process events with resolution and theory attribution
        for i, event_data in enumerate(event_data_list):
            try:
                # Calculate confidence
                confidence = self._calculate_calibrated_confidence(
                    event_data, logprobs_list[i]
                ) if enable_confidence_calibration else float(event_data.get("confidence", 0.7))
                
                # Validate event type against schema
                event_type = event_data.get("event_category", "OTHER")
                if not self.ontology.validate_event_type(event_type):
                    print(f"[warning] Invalid event type '{event_type}', using fallback")
                    event_type = "PHYSICAL_ACTION"  # Fallback
                
                # Infer theory from event type
                theory = self._infer_theory_from_event_type(event_type)
                
                # Process why_factors
                raw_why = event_data.get("why_factors", [])
                why_factors_list = [str(w) for w in raw_why] if isinstance(raw_why, list) else []
                
                seq = self.global_event_sequence
                self.global_event_sequence += 1
                
                # Create event with theory attribution
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
                        event.id, aid, actor_name, "actor", "PRODUCES_ACTOR", 1.0,
                        agent_type=None,  # Will be classified later if enabled
                        theory=theory
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
                        event.id, pid, pat_name, "patient", "PRODUCES_PATIENT", 1.0,
                        agent_type=None,
                        theory=theory
                    ))
                    entity_occurrences_batch[f"patient:{pat_name.lower()}"].append((event.id, seq))
                
                event.patients = clean_patients
                
                # Process why_factors
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

    async def _classify_agent_types(self, events: List[schemas.CEKEvent], 
                                    event_produces: List[schemas.EventProducesEntity]) -> Dict[str, str]:
        """
        OPTIONAL: Automatically classify agent types using LLM.
        Returns {agent_name: agent_type} mapping.
        """
        print("[agent_classification] Analyzing character roles...")
        
        # Group events by character
        character_events = defaultdict(list)
        for event in events:
            for actor in event.actors:
                character_events[actor].append(event.raw_description)
            for patient in event.patients:
                character_events[patient].append(event.raw_description)
        
        # Classify each character
        classifications = {}
        agent_type_names = self.ontology.get_agent_type_names()
        
        # FIX: Validate that we have agent types defined
        if not agent_type_names:
            print("[warning] No agent types defined, skipping classification")
            return classifications
        
        for char_name, char_events in character_events.items():
            if len(char_events) < 2:  # Skip minor characters
                continue
            
            # Sample events if too many
            sample_events = char_events[:10] if len(char_events) > 10 else char_events
            
            try:
                agent_type = await llm_service.classify_agent_type(
                    char_name, sample_events, agent_type_names,
                    self.openai_model, self.client
                )
                
                # FIX: Validate agent type before using
                if self.ontology.validate_agent_type(agent_type):
                    classifications[char_name] = agent_type
                    print(f"[agent_classification] {char_name} → {agent_type}")
                else:
                    print(f"[warning] Invalid agent type '{agent_type}' for {char_name}")
            except Exception as e:
                print(f"[warning] Failed to classify {char_name}: {e}")
        
        # Update EventProducesEntity with agent types
        for prod in event_produces:
            if prod.entity_type in ['actor', 'patient']:
                if prod.entity_name in classifications:
                    prod.agent_type = classifications[prod.entity_name]
        
        return classifications

    async def _batch_causal_linking_mixed_theory(self, events, entity_occurrences, 
                                                 window, sample_rate, batch_size,
                                                 enable_long_range=False,
                                                 theory_mode="mixed"):
        """
        Causal linking with multiple theory support.
        
        FIX: Added theory_mode parameter to support single-theory mode
        theory_mode: "mixed" (both), "mckee", "truby"
        """
        self.dag_validator.add_events(events)
        pairs_set = set()
        
        if enable_long_range:
            print("[long_range] Enabling cross-chapter inference...")
            # Long-range: consider events from entire narrative
            for i, ev in enumerate(events):
                # Look back much further for long-range connections
                lookback = min(50, i)  # Up to 50 events back
                start = max(0, i - lookback)
                
                for j in range(start, i):
                    if events[j].sequence < ev.sequence:
                        # Add all pairs, will sample later
                        pairs_set.add((events[j].id, ev.id))
        else:
            # Standard window-based (within chapter boundaries)
            for i, ev in enumerate(events):
                start = max(0, i - window)
                for j in range(start, i):
                    # Respect chapter boundaries unless long_range enabled
                    if events[j].sequence < ev.sequence and events[j].chapter == ev.chapter:
                        pairs_set.add((events[j].id, ev.id))
        
        # Entity co-occurrence pairs (works for both modes)
        for k, occs in entity_occurrences.items():
            if k.startswith("actor:") or k.startswith("patient:"):
                for i in range(len(occs) - 1):
                    c_id, c_seq = occs[i]
                    e_id, e_seq = occs[i + 1]
                    if c_seq < e_seq:
                        pairs_set.add((c_id, e_id))
        
        # Sample if too many pairs (for long-range mode)
        pairs_list = list(pairs_set)
        if enable_long_range and len(pairs_list) > 5000:
            import random
            random.shuffle(pairs_list)
            pairs_list = pairs_list[:5000]
            print(f"[long_range] Sampled {len(pairs_list)} pairs from {len(pairs_set)} candidates")
        else:
            print(f"[causal linking] Found {len(pairs_list)} candidate pairs.")
        
        ev_map = {e.id: e for e in events}
        pairs_with_text = [(ev_map[c].raw_description, ev_map[e].raw_description, c, e) 
                           for c, e in pairs_list if c in ev_map and e in ev_map]
        
        causal_links = []
        
        # FIX: Support multiple theory modes
        theories_to_process = []
        if theory_mode == "mixed":
            theories_to_process = [("mckee", self.mckee_relations), ("truby", self.truby_relations)]
        elif theory_mode == "mckee":
            theories_to_process = [("mckee", self.mckee_relations)]
        elif theory_mode == "truby":
            theories_to_process = [("truby", self.truby_relations)]
        else:
            print(f"[warning] Unknown theory_mode '{theory_mode}', defaulting to mckee")
            theories_to_process = [("mckee", self.mckee_relations)]
        
        for theory_name, relation_ontology in theories_to_process:
            if not relation_ontology:
                print(f"[warning] No relations defined for {theory_name}, skipping")
                continue
                
            print(f"\n[{theory_mode}] Analyzing with {theory_name.upper()} theory...")
            
            for i in range(0, len(pairs_with_text), batch_size):
                batch = pairs_with_text[i:i + batch_size]
                results = await llm_service.batch_assess_pairs(
                    batch, self.openai_model, self.client, relation_ontology
                )
                
                for (c_txt, e_txt, c_id, e_id), res in zip(batch, results):
                    if not res:
                        continue
                        
                    raw_rt = res.get("relationType")
                    if isinstance(raw_rt, list):
                        raw_rt = raw_rt[0] if raw_rt else None
                    
                    if raw_rt and str(raw_rt).upper() not in [None, "NONE", "NULL"]:
                        # Validate relation type against schema
                        rt_str = str(raw_rt).upper()
                        if not self.ontology.validate_relation_type(rt_str, theory_name):
                            continue
                        
                        # Get directionality from schema
                        directionality = self.ontology.get_relation_directionality(rt_str, theory_name)
                        
                        if self.dag_validator.add_edge(c_id, e_id):
                            # FIX: Use consistent theory naming
                            theory_tag = f"@{theory_name.capitalize()}"
                            
                            causal_links.append(schemas.CausalLink(
                                c_id, e_id, rt_str, res.get("mechanism", ""),
                                float(res.get("weight", 0)), float(res.get("confidence", 0)),
                                theory=theory_tag,
                                directionality=directionality
                            ))
                
                print(f"[{theory_name}] Processed {min(i + batch_size, len(pairs_with_text))}/{len(pairs_with_text)} pairs")
        
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

    async def _generate_scenes_with_all_entities(self, events, event_produces):
        """
        Generate scenes that contain ALL entities and events.
        Scene-centric structure: everything belongs to a scene.
        """
        print("[scenes] Creating scene-centric structure...")
        scenes = []
        by_chap = defaultdict(list)
        for e in events:
            by_chap[e.chapter].append(e)
        
        # Map event_id to entities
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
            
            # If no scenes detected, create default scene for chapter
            if not data:
                data = [{
                    "event_ids": [e.id for e in evs],
                    "theme": f"Chapter {cid} narrative",
                    "confidence": 0.5
                }]
            
            # FIX: Build event ID lookup for efficient checking
            event_ids_in_chapter = {e.id for e in evs}
            
            for d in data:
                try:
                    event_ids = d.get("event_ids", [])
                    
                    # FIX: Filter to only valid event IDs
                    valid_event_ids = [eid for eid in event_ids if eid in event_ids_in_chapter]
                    
                    # Aggregate ALL entities from included events
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
                    
                    # Combine all participants
                    all_participants = list(all_actors | all_patients)
                    
                    # Create scene with full entity data
                    scene = schemas.Scene(
                        utils._make_id("scene"), cid, valid_event_ids,
                        primary_location, time_period, all_participants,
                        d.get("theme", ""), "",
                        float(d.get("confidence", 0)),
                        place_type=None,
                        time_type=None
                    )
                    
                    # Store entity lists for later use
                    scene.all_actors = list(all_actors)
                    scene.all_patients = list(all_patients)
                    scene.all_whyfactors = list(all_whyfactors)
                    
                    scenes.append(scene)
                except Exception as e:
                    print(f"[warning] Failed to create scene: {e}")
        
        # Ensure all events are in at least one scene
        all_event_ids_in_scenes = set()
        for scene in scenes:
            all_event_ids_in_scenes.update(scene.included_event_ids)
        
        orphan_events = [e for e in events if e.id not in all_event_ids_in_scenes]
        if orphan_events:
            print(f"[scenes] Found {len(orphan_events)} orphan events, creating default scenes...")
            by_chapter = defaultdict(list)
            for ev in orphan_events:
                by_chapter[ev.chapter].append(ev)
            
            for cid, evs in by_chapter.items():
                event_ids = [e.id for e in evs]
                all_actors = set()
                all_patients = set()
                
                for ev in evs:
                    entities = event_to_entities[ev.id]
                    all_actors.update(entities["actors"])
                    all_patients.update(entities["patients"])
                
                scene = schemas.Scene(
                    utils._make_id("scene"), cid, event_ids,
                    None, None, list(all_actors | all_patients),
                    f"Chapter {cid} miscellaneous", "",
                    0.3, None, None
                )
                scene.all_actors = list(all_actors)
                scene.all_patients = list(all_patients)
                scene.all_whyfactors = []
                scenes.append(scene)
        
        print(f"[scenes] Generated {len(scenes)} scenes containing all events and entities")
        return scenes

    async def run_async(self, text_path, out_json, out_cypher, out_csv_dir,
                       max_chapters=None, batch_size=5, causal_window=10, 
                       causal_sample_rate=0.5, causal_batch_size=10, 
                       paragraph_chunk_size=1, extraction_style="detailed",
                       graph_model="star",
                       # NEW: Optional features
                       enable_mixed_theory=True,
                       enable_agent_classification=False,
                       enable_long_range_inference=False,
                       enable_scene_grouping=True,
                       enable_semantic_linking=False,
                       enable_llm_expansion=True,
                       enable_confidence_calibration=True):
        """
        Main pipeline execution with optional advanced features.
        
        FEATURES:
        - enable_mixed_theory: Use both McKee and Truby simultaneously (default: True)
        - enable_agent_classification: Automatically classify character roles (default: False)
        - enable_long_range_inference: Cross-chapter causal links (default: False)
        """
        # FIX: Determine theory mode
        theory_mode = "mixed" if enable_mixed_theory else "mckee"
        mode_desc = "Mixed Theory (McKee + Truby)" if enable_mixed_theory else "Single Theory (McKee)"
        
        print(f"[pipeline] Loading text...")
        print(f"[pipeline] Graph Model: {graph_model}")
        print(f"[pipeline] Causal Mode: {mode_desc}")
        print(f"[pipeline] Agent Classification: {'ON' if enable_agent_classification else 'OFF'}")
        print(f"[pipeline] Long-Range Inference: {'ON' if enable_long_range_inference else 'OFF'}")
        
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
        all_produces.extend(new_prods