"""
Integrated Pipeline - Smart Causal Linking + Semantic Analysis
Combines optimized_linking.py and integrated_semantic.py for best results
Now with configurable chunking for granularity control
"""

import asyncio
import traceback
import os
import json
import random
import textwrap
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple

from . import config, schemas, utils, text_processor, llm_service, graph_builder, graph_mapper, exporters
from .ontology_loader import get_ontology_manager, OntologyManager
from .optimized_linking import intelligent_long_range_linking
from .integrated_semantic import (
    process_pairs_with_semantic_linking,
    create_hybrid_semantic_links
)
# --- FIX: IMPORT RESOLVER ---
from .coreference_resolver import get_resolver

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
        self.ontology = get_ontology_manager(schema_path)
        self.event_ontology = self.ontology.get_event_type_names()
        self.mckee_relations = self.ontology.get_relation_type_names(THEORY_MCKEE_LOWER)
        self.truby_relations = self.ontology.get_relation_type_names(THEORY_TRUBY_LOWER)
        
        # Initialize resolver
        self.resolver = get_resolver()
        
        print(f"[pipeline] Initialized with {len(self.event_ontology)} event types")
        print(f"[pipeline] McKee relations: {len(self.mckee_relations)}")
        print(f"[pipeline] Truby relations: {len(self.truby_relations)}")
    
    def _calculate_calibrated_confidence(self, event_data, logprobs):
        """Simplified confidence calculation"""
        p_llm = float(event_data.get("confidence", 0.7))
        score = 1.0
        if not event_data.get("actors"): score -= 0.2
        if not event_data.get("location_context"): score -= 0.1
        return round((0.6 * p_llm) + (0.4 * score), 4)

    def _infer_theory_from_event_type(self, event_type: str) -> str:
        if event_type in self.ontology.event_types:
            return normalize_theory_name(self.ontology.event_types[event_type].theory)
        return THEORY_MCKEE

    def _parse_event_json_data(self, event_data_list, chapter_id, logprobs, 
                               enable_confidence_calibration, graph_model="star"):
        """Parse event data without coreference resolution (LLM handles this)"""
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
                
                # --- FIX: USE RESOLVER TO FILTER PRONOUNS ---
                
                # Direct actor extraction with filtering
                raw_actors = event_data.get("actors", [])
                if isinstance(raw_actors, str):
                    raw_actors = [raw_actors]
                
                clean_actors = []
                for actor in raw_actors:
                    if isinstance(actor, str) and len(actor) > 1:
                        actor_name = actor.strip()
                        # Strict pronoun/generic filtering
                        if not self.resolver.is_valid_character_name(actor_name):
                            continue
                            
                        clean_actors.append(actor_name)
                        aid = graph_builder._generate_entity_id(actor_name, "agent", event.id, graph_model)
                        all_produces.append(schemas.EventProducesEntity(
                            event.id, aid, actor_name, "actor", "PRODUCES_ACTOR", 1.0,
                            agent_type=None, theory=theory
                        ))
                        entity_occurrences_batch[f"actor:{actor_name.lower()}"].append((event.id, seq))
                
                event.actors = clean_actors

                # Direct patient extraction with filtering
                raw_patients = event_data.get("patients", [])
                if isinstance(raw_patients, str):
                    raw_patients = [raw_patients]
                
                clean_patients = []
                for patient in raw_patients:
                    if isinstance(patient, str) and len(patient) > 1:
                        pat_name = patient.strip()
                        # Strict pronoun/generic filtering
                        if not self.resolver.is_valid_character_name(pat_name):
                            continue

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

    async def _process_chapter_chunked(self, chapter_text, chapter_id, 
                                      enable_confidence_calibration, 
                                      extraction_style, graph_model,
                                      chunk_size=3000):
        """
        HYBRID APPROACH: Process chapter in smart chunks
        - Splits chapter into ~chunk_size char chunks (configurable)
        - Processes chunks concurrently
        - Maintains granularity while reducing API calls
        """
        try:
            # Split chapter into overlapping chunks to maintain context
            chunks = []
            sentences = chapter_text.split('. ')
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip() + '. '
                if current_length + len(sentence) > chunk_size and current_chunk:
                    chunks.append(''.join(current_chunk))
                    # Keep last 2 sentences for context overlap
                    current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else []
                    current_length = sum(len(s) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_length += len(sentence)
            
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            print(f"[chapter {chapter_id}] Processing {len(chunks)} chunks ({len(chapter_text)} chars total)")
            
            # Process chunks concurrently (reduces total time)
            tasks = []
            for i, chunk in enumerate(chunks):
                tasks.append(llm_service.extract_events_from_text(
                    chunk, f"{chapter_id}.{i}", self.openai_model, self.client,
                    False, False, extraction_style, self.event_ontology
                ))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge all events from chunks
            all_events = []
            all_produces = []
            entity_occurrences_batch = defaultdict(list)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"[warning] Chunk processing failed: {result}")
                    continue
                
                data, logprobs = result
                events, produces, occurrences = self._parse_event_json_data(
                    data, chapter_id, logprobs, enable_confidence_calibration, graph_model
                )
                
                all_events.extend(events)
                all_produces.extend(produces)
                for k, v in occurrences.items():
                    entity_occurrences_batch[k].extend(v)
            
            print(f"[chapter {chapter_id}] Extracted {len(all_events)} events from {len(chunks)} chunks")
            return all_events, all_produces, entity_occurrences_batch
            
        except Exception as e:
            print(f"[error] Chapter processing failed: {e}")
            return [], [], defaultdict(list)

    async def _classify_agent_types(self, events: List[schemas.CEKEvent], 
                                    event_produces: List[schemas.EventProducesEntity]) -> Dict[str, str]:
        """Agent classification with cheaper model"""
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

    async def _integrated_causal_and_semantic_linking(
        self, 
        events, 
        entity_occurrences, 
        theory_mode="mixed",
        max_concurrent_calls=10,
        max_pairs=5000,
        enable_semantic=True
    ):
        """
        INTEGRATED: Smart causal linking + semantic analysis in ONE pass
        Uses optimized_linking.py for intelligence + integrated_semantic.py for dual extraction
        """
        print(f"\n{'='*60}")
        print(f"INTEGRATED CAUSAL + SEMANTIC ANALYSIS")
        print(f"{'='*60}")
        
        self.dag_validator.add_events(events)
        all_causal_links = []
        all_semantic_links = []
        mckee_link_count = 0
        truby_link_count = 0
        
        # Determine theories to process
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
        
        # Import the smart linker
        from .optimized_linking import IntelligentCausalLinker
        linker = IntelligentCausalLinker(use_embeddings=True)
        
        # Get smart candidate pairs (replaces aggressive filtering)
        print(f"\n[smart_linking] Generating intelligent candidate pairs...")
        candidate_pairs = linker.get_candidate_pairs(
            events, entity_occurrences, max_pairs
        )
        
        if not candidate_pairs:
            print("[warning] No candidate pairs found")
            return [], [], 0, 0
        
        # Convert to text format for assessment
        event_map = {e.id: e for e in events}
        pairs_with_text = []
        
        for cause_id, effect_id in candidate_pairs:
            if cause_id in event_map and effect_id in event_map:
                cause = event_map[cause_id]
                effect = event_map[effect_id]
                
                # Truncate for efficiency
                cause_text = cause.raw_description[:150]
                effect_text = effect.raw_description[:150]
                
                pairs_with_text.append((cause_text, effect_text, cause_id, effect_id))
        
        print(f"[smart_linking] Selected {len(pairs_with_text):,} high-quality pairs")
        
        # Process each theory with integrated semantic analysis
        for theory_name, relation_ontology, theory_tag in theories_to_process:
            if not relation_ontology:
                continue
            
            print(f"\n{'='*60}")
            print(f"THEORY: {theory_tag}")
            print(f"{'='*60}")
            
            if enable_semantic:
                # INTEGRATED: Both causal and semantic in ONE API call
                print(f"[{theory_name}] Using INTEGRATED assessment (causal + semantic)")
                
                causal_links, semantic_links = await process_pairs_with_semantic_linking(
                    pairs_with_text,
                    self.openai_model,
                    self.client,
                    relation_ontology,
                    theory_name,
                    self.dag_validator,
                    self.ontology,
                    llm_service._async_llm_json_call,
                    max_concurrent_calls,
                    bulk_size=50
                )
                
                all_causal_links.extend(causal_links)
                all_semantic_links.extend(semantic_links)
                
                if theory_tag == THEORY_MCKEE:
                    mckee_link_count += len(causal_links)
                elif theory_tag == THEORY_TRUBY:
                    truby_link_count += len(causal_links)
                
            else:
                # Standard causal-only processing
                print(f"[{theory_name}] Using standard causal assessment")
                
                results = []
                BULK_SIZE = 50
                
                for i in range(0, len(pairs_with_text), BULK_SIZE * max_concurrent_calls):
                    batch_end = min(i + BULK_SIZE * max_concurrent_calls, len(pairs_with_text))
                    
                    chunks = []
                    for j in range(i, batch_end, BULK_SIZE):
                        chunk = pairs_with_text[j:j + BULK_SIZE]
                        chunks.append(llm_service.assess_pairs_bulk(
                            chunk, self.openai_model, self.client, relation_ontology
                        ))
                    
                    chunk_results = await asyncio.gather(*chunks)
                    
                    for chunk_result in chunk_results:
                        results.extend(chunk_result)
                    
                    if (i // (BULK_SIZE * max_concurrent_calls)) % 5 == 0:
                        progress = 100 * len(results) / len(pairs_with_text)
                        print(f"[progress] {len(results):,}/{len(pairs_with_text):,} ({progress:.1f}%)")
                
                # Create causal links
                for pair, result in zip(pairs_with_text, results):
                    if not result:
                        continue
                    
                    _, _, cause_id, effect_id = pair
                    
                    rel_type = result.get("relationType")
                    if not rel_type or str(rel_type).upper() in ["NONE", "NULL"]:
                        continue
                    
                    rt_str = str(rel_type).upper()
                    
                    if not self.ontology.validate_relation_type(rt_str, theory_name):
                        continue
                    
                    directionality = self.ontology.get_relation_directionality(rt_str, theory_name)
                    
                    if self.dag_validator.add_edge(cause_id, effect_id):
                        all_causal_links.append(schemas.CausalLink(
                            cause_id, effect_id, rt_str, result.get("mechanism", ""),
                            float(result.get("weight", 0)),
                            float(result.get("confidence", 0)),
                            theory=theory_tag,
                            directionality=directionality
                        ))
                        
                        if theory_tag == THEORY_MCKEE:
                            mckee_link_count += 1
                        elif theory_tag == THEORY_TRUBY:
                            truby_link_count += 1
        
        # Add local embedding-based semantic links (high quality, zero cost)
        if enable_semantic:
            print(f"\n[semantic] Adding local embedding-based thematic links...")
            all_semantic_links = create_hybrid_semantic_links(events, all_semantic_links)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Pairs Evaluated: {len(pairs_with_text):,}")
        print(f"Causal Links: {len(all_causal_links):,}")
        if theory_mode == "mixed":
            print(f"  - McKee: {mckee_link_count:,}")
            print(f"  - Truby: {truby_link_count:,}")
        if enable_semantic:
            print(f"Semantic Links: {len(all_semantic_links):,}")
        print(f"{'='*60}\n")
        
        return all_causal_links, all_semantic_links, mckee_link_count, truby_link_count

    async def _generate_scenes_optimized(self, events, event_produces):
        """Scene generation with cheaper model"""
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
        
        print(f"[scenes] Generated {len(scenes)} scenes")
        return scenes

    async def run_async(self, text_path, out_json, out_cypher, out_csv_dir,
                   max_chapters=None,
                   graph_model="star",
                   enable_mixed_theory=True,
                   enable_agent_classification=False,
                   enable_scene_grouping=True,
                   enable_confidence_calibration=True,
                   enable_semantic_linking=True,
                   max_concurrent_calls=10,
                   max_long_range_pairs=5000,
                   chunk_size=3000):
        """
        INTEGRATED PIPELINE: Smart causal linking + semantic analysis
        Now with configurable chunk_size for granularity control
        """
        
        theory_mode = "mixed" if enable_mixed_theory else THEORY_MCKEE_LOWER
        mode_desc = "Mixed Theory" if enable_mixed_theory else "Single Theory"
        
        print(f"\n{'='*60}")
        print(f"INTEGRATED CEKG PIPELINE")
        print(f"Smart Linking + Semantic Analysis")
        print(f"{'='*60}")
        print(f"[pipeline] Graph Model: {graph_model}")
        print(f"[pipeline] Theory Mode: {mode_desc}")
        print(f"[pipeline] Chunk Size: {chunk_size} chars (~{chunk_size//800} paragraphs)")
        print(f"[pipeline] Max Causal Pairs: {max_long_range_pairs:,}")
        print(f"[pipeline] Semantic Linking: {'✓' if enable_semantic_linking else '✗'}")
        print(f"[pipeline] Processing Mode: CHUNKED (concurrent chunks per chapter)")
        print(f"{'='*60}\n")
        
        raw = text_processor.load_text(text_path)
        chapters = text_processor.split_chapters(raw)[:max_chapters] if max_chapters else text_processor.split_chapters(raw)
        
        all_events, all_produces, entity_occurrences = [], [], defaultdict(list)
        self.global_event_sequence = 0

        # Process each chapter with chunking
        for cid, txt in chapters:
            print(f"\n[chapter {cid}] Processing with smart chunking...")
            e, p, o = await self._process_chapter_chunked(
                txt, cid, enable_confidence_calibration, "detailed", graph_model,
                chunk_size=chunk_size
            )
            all_events.extend(e)
            all_produces.extend(p)
            for k, v in o.items():
                entity_occurrences[k].extend(v)
            print(f"[chapter {cid}] Extracted {len(e)} events")

        print(f"\n[pipeline] Total events extracted: {len(all_events)}")
        
        # Context propagation
        all_events = graph_builder.propagate_context_attributes(all_events)
        new_prods, entity_occurrences = graph_builder.propagate_context(
            all_events, all_produces, entity_occurrences, graph_model=graph_model
        )
        all_produces.extend(new_prods)
        
        entity_to_event_links = graph_builder.create_entity_to_event_links(
            entity_occurrences, all_produces, graph_model=graph_model
        )
        
        # Agent classification (optional)
        agent_classifications = {}
        if enable_agent_classification:
            agent_classifications = await self._classify_agent_types(all_events, all_produces)
        
        # INTEGRATED: Causal + Semantic Linking
        causal_links, semantic_links, mckee_count, truby_count = await self._integrated_causal_and_semantic_linking(
            all_events, 
            entity_occurrences, 
            theory_mode, 
            max_concurrent_calls, 
            max_long_range_pairs,
            enable_semantic_linking
        )
        
        # Scene grouping (optional)
        scenes = []
        if enable_scene_grouping:
            scenes = await self._generate_scenes_optimized(all_events, all_produces)
        
        all_characters = set()
        for prod in all_produces:
            if prod.entity_type in ['actor', 'patient']:
                all_characters.add(prod.entity_name)
        
        print(f"\n{'='*60}")
        print(f"EXPORTING RESULTS")
        print(f"{'='*60}")
        
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
        is_valid_dag = self.dag_validator.validate_dag()
        
        cache_stats = await llm_service.get_cache_sizes()
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Events: {len(all_events):,}")
        print(f"Characters: {len(all_characters):,}")
        print(f"Causal Links: {len(causal_links):,}")
        if enable_mixed_theory:
            print(f"  - McKee: {mckee_count:,}")
            print(f"  - Truby: {truby_count:,}")
        if enable_semantic_linking:
            print(f"Semantic Links: {len(semantic_links):,}")
        print(f"Scenes: {len(scenes):,}")
        print(f"DAG Valid: {is_valid_dag}")
        print(f"{'='*60}\n")

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