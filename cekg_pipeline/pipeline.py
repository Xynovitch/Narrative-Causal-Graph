import asyncio
import traceback
import random
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple

# Import from our new modules
from . import config
from . import schemas
from . import utils
from . import text_processor
from . import llm_service
from . import graph_builder  # <-- FIX: This import was missing
from . import graph_mapper
from . import exporters

class CEKGPreprocessor:
    
    def __init__(self, openai_model: Optional[str] = None):
        if not config.OPENAI_API_KEY:
             raise RuntimeError("OPENAI_API_KEY not set. Please set the environment variable.")
        
        self.openai_model = openai_model or config.OPENAI_MODEL
        self.client = llm_service.init_openai_client(config.OPENAI_API_KEY)
        self.dag_validator = utils.DAGValidator()
        self.global_event_sequence = 0
        
    async def _process_event_batch(
        self, 
        paragraphs_with_chapter: List[tuple[str, int]]
    ) -> Tuple[List[schemas.CEKEvent], List[schemas.EventProducesEntity], Dict[str, List[Tuple[str, int]]]]:
        """
        Process a batch of paragraphs, return parsed objects.
        This is where the logic from your old `process_chapter_events` lives.
        """
        all_events = []
        all_produces = []
        entity_occurrences_batch = defaultdict(list)

        # 1. Get JSON data from LLM service
        try:
            events_batch_json = await llm_service.batch_extract_events(
                paragraphs_with_chapter, 
                self.openai_model,
                self.client
            )
        except Exception as e:
            print(f"[error] Batch processing failed: {e}")
            traceback.print_exc()
            return [], [], {} # Return empty on total batch failure

        # 2. Parse JSON into schema objects
        for (para, chapter_id), para_events_json in zip(paragraphs_with_chapter, events_batch_json):
            if isinstance(para_events_json, Exception) or not para_events_json:
                continue
                
            for event_data in para_events_json:
                try:
                    why_factors = utils._normalize_weights(event_data.get("whyFactors", []))
                    if not why_factors:
                        # LLM failed to provide mandatory whyFactor
                        print(f"[warning] LLM failed to provide mandatory whyFactor for event: {event_data.get('name')}")
                        why_factors = [{"factor": "narrative progression", "weight": 1.0, "category": "Contextual Cause"}]

                    cause_weight = sum(wf.get("weight", 0) for wf in why_factors)
                    
                    action_type = event_data.get("actionType", "")
                    action_type = config.CONTROLLED_ACTION_ONTOLOGY.get(action_type.lower(), action_type)
                    
                    raw_location = event_data.get("location")
                    location = raw_location.strip() if raw_location else None
                    location_id = f"place/{location.lower().replace(' ', '_')}" if location else None
                    
                    event_sequence = self.global_event_sequence
                    self.global_event_sequence += 1
                    
                    event = schemas.CEKEvent(
                        id=utils._make_id("event"),
                        name=event_data.get("name", ""),
                        eventType=event_data.get("eventType", "other"),
                        actionType=action_type,
                        source_quote=event_data.get("quote", ""),
                        time=event_data.get("time"),
                        location=location,
                        location_id=location_id,
                        causeWeight=cause_weight,
                        confidence=float(event_data.get("confidence", 0.7)),
                        chapter=chapter_id, # Use the chapter_id passed in
                        sequence=event_sequence
                    )
                    all_events.append(event)
                    
                    # --- Process Entities ---
                    
                    # Actors
                    actors = event_data.get("actors", [])
                    if not actors and event_data.get("actor"): # Handle fallback
                         actors = [{"name": event_data["actor"], "strength": 1.0}]
                    
                    for actor in actors:
                        raw_actor_name = actor.get("name")
                        if raw_actor_name:
                            actor_name = raw_actor_name.strip()
                            actor_key = actor_name.lower()
                            actor_id = utils._make_id(f"agent_{actor_key.replace(' ', '_')}")
                            
                            all_produces.append(schemas.EventProducesEntity(
                                event_id=event.id, entity_id=actor_id,
                                entity_name=actor_name, entity_type="actor",
                                relationship="PRODUCES_ACTOR",
                                strength=float(actor.get("strength", 1.0))
                            ))
                            entity_occurrences_batch[f"actor:{actor_key}"].append((event.id, event.sequence))
                    
                    # Patients
                    patients = event_data.get("patients", [])
                    if not patients and event_data.get("patient"): # Handle fallback
                        patients = [{"name": event_data["patient"], "strength": 1.0}]

                    for patient in patients:
                        raw_patient_name = patient.get("name")
                        if raw_patient_name:
                            patient_name = raw_patient_name.strip()
                            patient_key = patient_name.lower()
                            patient_id = utils._make_id(f"agent_{patient_key.replace(' ', '_')}")
                            
                            all_produces.append(schemas.EventProducesEntity(
                                event_id=event.id, entity_id=patient_id,
                                entity_name=patient_name, entity_type="patient",
                                relationship="PRODUCES_PATIENT",
                                strength=float(patient.get("strength", 1.0))
                            ))
                            entity_occurrences_batch[f"patient:{patient_key}"].append((event.id, event.sequence))

                    # WhyFactors
                    for wf in why_factors:
                        raw_factor = wf.get('factor')
                        if raw_factor:
                            factor_name = raw_factor.strip()
                            factor_key = factor_name.lower()
                            wf_id = utils._make_id(f"why_{factor_key[:30].replace(' ', '_')}")
                            
                            all_produces.append(schemas.EventProducesEntity(
                                event_id=event.id, entity_id=wf_id,
                                entity_name=factor_name, entity_type="whyfactor",
                                relationship="PRODUCES_MOTIVATION",
                                strength=wf.get('weight', 0.5) # Use weight as strength
                            ))
                            entity_occurrences_batch[f"whyfactor:{factor_key}"].append((event.id, event.sequence))

                    # Location
                    if location:
                        location_key = location.lower()
                        place_id = utils._make_id(f"place_{location_key.replace(' ', '_')}")
                        
                        all_produces.append(schemas.EventProducesEntity(
                            event_id=event.id, entity_id=place_id,
                            entity_name=location, entity_type="place",
                            relationship="PRODUCES_LOCATION", strength=0.8
                        ))
                        entity_occurrences_batch[f"place:{location_key}"].append((event.id, event.sequence))
                    
                except Exception as e:
                    print(f"[warning] Failed to process single event: {event_data.get('name')}. Error: {e}")
                    traceback.print_exc()
                    continue
        
        return all_events, all_produces, entity_occurrences_batch

    async def _batch_causal_linking(
        self,
        events: List[schemas.CEKEvent],
        entity_occurrences: Dict[str, List[Tuple[str, int]]],
        window: int, 
        sample_rate: float, 
        batch_size: int
    ) -> List[schemas.CausalLink]:
        
        self.dag_validator.add_events(events)
        print(f"[dag] Initialized DAG validator with {len(events)} events")
        
        pairs_to_assess_set: Set[tuple[str, str]] = set()
        causal_links = []

        # --- PASS 1: LOCAL WINDOW ---
        print("[causal linking] Pass 1/3: Assessing local window...")
        for i, ev in enumerate(events):
            start = max(0, i - window)
            for j in range(start, i):
                cand = events[j]
                if cand.sequence >= ev.sequence:
                    continue
                pairs_to_assess_set.add((cand.id, ev.id))

        # --- PASS 2: SHARED ACTOR/PATIENT ---
        print("[causal linking] Pass 2/3: Assessing shared actors...")
        for entity_key, occurrences in entity_occurrences.items():
            if not (entity_key.startswith("actor:") or entity_key.startswith("patient:")):
                continue
            for i in range(len(occurrences) - 1):
                cause_event_id, _ = occurrences[i]
                effect_event_id, _ = occurrences[i+1]
                cause_seq = self.dag_validator.event_sequence_map.get(cause_event_id)
                effect_seq = self.dag_validator.event_sequence_map.get(effect_event_id)
                if cause_seq is not None and effect_seq is not None and cause_seq < effect_seq:
                    pairs_to_assess_set.add((cause_event_id, effect_event_id))

        # --- PASS 3: SHARED LOCATION ---
        print("[causal linking] Pass 3/3: Assessing shared locations...")
        for entity_key, occurrences in entity_occurrences.items():
            if not entity_key.startswith("place:"):
                continue
            for i in range(len(occurrences) - 1):
                cause_event_id, _ = occurrences[i]
                effect_event_id, _ = occurrences[i+1]
                cause_seq = self.dag_validator.event_sequence_map.get(cause_event_id)
                effect_seq = self.dag_validator.event_sequence_map.get(effect_event_id)
                if cause_seq is not None and effect_seq is not None and cause_seq < effect_seq:
                    pairs_to_assess_set.add((cause_event_id, effect_event_id))

        print(f"[causal linking] Found {len(pairs_to_assess_set)} unique candidate pairs.")
        
        event_map_by_id = {ev.id: ev for ev in events}
        
        pairs_to_assess = []
        for cause_id, effect_id in pairs_to_assess_set:
            cause_ev = event_map_by_id.get(cause_id)
            effect_ev = event_map_by_id.get(effect_id)
            
            if cause_ev and effect_ev:
                pairs_to_assess.append((
                    cause_ev.source_quote, effect_ev.source_quote, cause_id, effect_id
                ))
        
        # Optional: Sampling
        # if len(pairs_to_assess) > (len(events) * 10): # Simple heuristic
        #     pairs_to_assess = random.sample(pairs_to_assess, int(len(pairs_to_assess) * sample_rate))
        
        print(f"[causal linking] Assessing {len(pairs_to_assess)} final pairs via LLM...")
        
        accepted_edges = 0
        rejected_edges = 0
        
        for batch_start in range(0, len(pairs_to_assess), batch_size):
            batch = pairs_to_assess[batch_start:batch_start + batch_size]
            
            assessments = await llm_service.batch_assess_pairs(
                batch, self.openai_model, self.client
            )
            
            for (cause_quote, effect_quote, cause_id, effect_id), assessment in zip(batch, assessments):
                if assessment is None or assessment.get("relationType") == "none":
                    continue
                
                if not self.dag_validator.add_edge(cause_id, effect_id):
                    # This rejects cycles or duplicate edges
                    rejected_edges += 1
                    continue
                
                cause_seq = self.dag_validator.event_sequence_map.get(cause_id)
                effect_seq = self.dag_validator.event_sequence_map.get(effect_id)
                
                if cause_seq is None or effect_seq is None:
                    # This should not happen if add_edge succeeded, but good to check
                    rejected_edges += 1
                    continue
                
                link = schemas.CausalLink(
                    cause_id=cause_id,
                    effect_id=effect_id,
                    relationType=assessment.get("relationType", "causes"),
                    mechanism=assessment.get("mechanism", ""),
                    sign=assessment.get("sign", "0"),
                    weight=float(assessment.get("weight", 0.0)),
                    confidence=float(assessment.get("confidence", 0.0)),
                    cause_sequence=cause_seq,
                    effect_sequence=effect_seq
                )
                causal_links.append(link)
                accepted_edges += 1
            
            print(f"[causal] Progress: {min(batch_start + batch_size, len(pairs_to_assess))}/{len(pairs_to_assess)} "
                  f"(accepted: {accepted_edges}, rejected: {rejected_edges})")
        
        print(f"[dag] Final edge statistics: {accepted_edges} accepted, {rejected_edges} rejected")
        stats = self.dag_validator.get_stats()
        print(f"[dag] ✓ DAG validated: {stats['nodes']} nodes, {stats['edges']} edges")
        print(f"[dag] Max in-degree: {stats['max_in_degree']}, Max out-degree: {stats['max_out_degree']}")
        
        return causal_links

    async def run_async(self, text_path: str, 
                       out_json: str,
                       out_cypher: str,
                       out_csv_dir: str,
                       max_chapters: Optional[int],
                       batch_size: int,
                       causal_window: int,
                       causal_sample_rate: float,
                       causal_batch_size: int) -> Dict[str, Any]:
        """Main async pipeline orchestrator"""
        
        print("[pipeline] Loading text...")
        raw = text_processor.load_text(text_path)
        chapters = text_processor.split_chapters(raw)

        if max_chapters is not None:
            chapters = chapters[:max_chapters]

        print(f"[pipeline] Processing {len(chapters)} chapters")
        print(f"[pipeline] DUAL FLOW: Event→Entity→Event chains ✓")

        # --- Store all data in local lists ---
        all_events: List[schemas.CEKEvent] = []
        all_produces: List[schemas.EventProducesEntity] = []
        entity_occurrences = defaultdict(list)
        self.global_event_sequence = 0 # Reset sequence counter

        for chapter_id, chapter_text in chapters:
            paragraphs = text_processor.split_into_paragraphs(chapter_text)
            print(f"[chapter {chapter_id}] Processing {len(paragraphs)} paragraphs...")
            
            for i in range(0, len(paragraphs), batch_size):
                batch_paras = paragraphs[i:i+batch_size]
                batch_with_chapter = [(para, chapter_id) for para in batch_paras]
                
                try:
                    events, produces, occ_batch = await self._process_event_batch(batch_with_chapter)
                    
                    all_events.extend(events)
                    all_produces.extend(produces)
                    for key, val_list in occ_batch.items():
                        entity_occurrences[key].extend(val_list)
                    
                    print(f"[chapter {chapter_id}] Batch {i//batch_size + 1}: extracted {len(events)} events")
                
                except Exception as e:
                    print(f"[error] Failed processing batch {i//batch_size + 1} in chapter {chapter_id}: {e}")
                    traceback.print_exc()
                    continue # Try next batch

        if not all_events:
            raise schemas.CEKGError("No events were extracted")

        print(f"[pipeline] Total events extracted: {len(all_events)}")
        
        # Events are already sorted by sequence number from the batch processor

        # --- FIX: REINSTATE THE CALLS TO GRAPH_BUILDER ---
        # Run Pass 2: Propagate context
        # Note: propagate_context mutates all_events in place, but returns new lists
        newly_produced, entity_occurrences = graph_builder.propagate_context(
            all_events, all_produces, entity_occurrences
        )
        all_produces.extend(newly_produced)
        print(f"[pipeline] Total event->entity productions after context: {len(all_produces)}")

        # Create entity→event links
        entity_to_event_links = graph_builder.create_entity_to_event_links(
            entity_occurrences, all_produces
        )
        # --- END FIX ---

        # Causal linking
        print(f"[pipeline] Starting causal linking")
        causal_links = []
        try:
            causal_links = await self._batch_causal_linking(
                events=all_events,
                entity_occurrences=entity_occurrences,
                window=causal_window,
                sample_rate=causal_sample_rate,
                batch_size=causal_batch_size
            )
        except Exception as e:
            print(f"[warning] Causal linking failed: {e}")
            traceback.print_exc()
        
        # Export results
        print(f"[pipeline] Mapping data to generic graph...")
        csvs = {}
        try:
            # 1. Map to generic graph structure
            generic_nodes, generic_rels = graph_mapper.map_to_generic_graph(
                all_events, all_produces, entity_to_event_links, causal_links
            )
            
            print(f"[pipeline] Exporting results...")
            
            # 2. Export JSON-LD (uses old lists)
            json_data = exporters.build_jsonld(
                all_events, all_produces, entity_to_event_links, causal_links
            )
            exporters.export_json(out_json, json_data)
            
            # 3. Export Cypher (uses new generic lists)
            exporters.export_neo4j_cypher(
                out_cypher, generic_nodes, generic_rels
            )
            
            # 4. Export CSV (uses old lists)
            csvs = exporters.export_csv(
                out_csv_dir, all_events, all_produces, entity_to_event_links, causal_links
            )
        except Exception as e:
            print(f"[error] Export failed: {e}")
            traceback.print_exc()
            # Don't raise, just report stats

        dag_stats = self.dag_validator.get_stats()
        cache_stats = await llm_service.get_cache_sizes()

        return {
            "json": out_json, 
            "cypher": out_cypher, 
            "csv": csvs,
            "stats": {
                "events": len(all_events),
                "event_produces_entity": len(all_produces),
                "entity_points_to_event": len(entity_to_event_links),
                "causal_links": len(causal_links),
                "cached_event_extractions": cache_stats["event_extraction_cache_size"],
                "cached_causal_assessments": cache_stats["assessment_cache_size"],
                "chapters_processed": len(set(ev.chapter for ev in all_events)),
                "dag_stats": dag_stats
            }
        }