import asyncio
import traceback
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any, Set, Tuple

# Import from our new modules
from . import config
from . import schemas
from . import utils
from . import text_processor
from . import llm_service
from . import graph_builder
from . import graph_mapper
from . import exporters

# Try to import sentence-transformers, but don't fail if it's not needed
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    SENTENCE_TRANSFORMER_MODEL = None
    print("[warning] 'sentence-transformers' library not found.")
    print("[warning] Calibrated Confidence (Item #3) will not be available.")


class CEKGPreprocessor:
    
    def __init__(self, openai_model: Optional[str] = None):
        if not config.OPENAI_API_KEY:
             raise RuntimeError("OPENAI_API_KEY not set. Please set the environment variable.")
        
        self.openai_model = openai_model or config.OPENAI_MODEL
        self.client = llm_service.init_openai_client(config.OPENAI_API_KEY)
        self.dag_validator = utils.DAGValidator()
        self.global_event_sequence = 0
    
    def _calculate_calibrated_confidence(
        self,
        event_data: Dict[str, Any],
        logprobs: Optional[Any] # This is a complex object from OpenAI
    ) -> float:
        """
        (EXPERIMENTAL) Calculate a calibrated confidence score.
        This is a placeholder for the complex logic from Item #3.
        """
        # 1. pLLM (Get logprob score)
        # This is complex. You would need to parse the `logprobs` object
        # to find the average token probability for the generated JSON.
        # For now, we'll just use the LLM's own confidence.
        p_llm = float(event_data.get("confidence", 0.7))

        # 2. plexical (Score JSON completeness)
        score = 1.0
        if not event_data.get("actors"): score -= 0.2
        if not event_data.get("whyFactors"): score -= 0.2 # We made this mandatory
        if not event_data.get("location"): score -= 0.1
        p_lexical = score

        # 3. pcontextual (Semantic similarity)
        p_contextual = 0.5 # Default
        if SENTENCE_TRANSFORMER_MODEL:
            try:
                name_embedding = SENTENCE_TRANSFORMER_MODEL.encode(event_data.get("name", ""), convert_to_tensor=True)
                quote_embedding = SENTENCE_TRANSFORMER_MODEL.encode(event_data.get("quote", ""), convert_to_tensor=True)
                sim = util.pytorch_cos_sim(name_embedding, quote_embedding)
                p_contextual = max(0, sim.item()) # Ensure non-negative
            except Exception as e:
                print(f"[warning] Could not calculate pcontextual: {e}")
        
        # Formula: confidence = α·pLLM + β·plexical + γ·pcontextual
        # These weights (α, β, γ) should be calibrated.
        alpha, beta, gamma = 0.4, 0.4, 0.2
        
        calibrated_score = (alpha * p_llm) + (beta * p_lexical) + (gamma * p_contextual)
        
        return round(calibrated_score, 4)

    def _parse_event_json_data(
        self,
        event_data_list: List[Dict[str, Any]],
        chapter_id: int,
        logprobs: Optional[Any], # logprobs object for the *entire* batch
        enable_confidence_calibration: bool
    ) -> Tuple[List[schemas.CEKEvent], List[schemas.EventProducesEntity], Dict[str, List[Tuple[str, int]]]]:
        """
        Helper function to parse a list of event JSON objects
        into schema objects and entity occurrences.
        """
        all_events = []
        all_produces = []
        entity_occurrences_batch = defaultdict(list)
        
        if not isinstance(event_data_list, list):
            event_data_list = []
        
        # Ensure logprobs_list has the same length as event_data_list
        if not logprobs:
            logprobs_list = [None] * len(event_data_list)
        else:
            # This is a simplification; proper logprob parsing is complex
            logprobs_list = [logprobs] * len(event_data_list)
        
        for i, event_data in enumerate(event_data_list):
            try:
                # --- Confidence Calculation ---
                if enable_confidence_calibration:
                    event_logprobs = logprobs_list[i] # Simplified
                    confidence = self._calculate_calibrated_confidence(event_data, event_logprobs)
                else:
                    confidence = float(event_data.get("confidence", 0.7))
                # ---
                
                why_factors = utils._normalize_weights(event_data.get("whyFactors", []))
                if not why_factors:
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
                    confidence=confidence, # Use the calculated confidence
                    chapter=chapter_id,
                    sequence=event_sequence
                )
                all_events.append(event)
                
                # --- Process Entities ---
                
                # Actors
                actors = event_data.get("actors", [])
                if not actors and event_data.get("actor"):
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
                if not patients and event_data.get("patient"):
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
                            strength=wf.get('weight', 0.5)
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

    async def _process_parallel_batch(
        self, 
        paragraphs_with_chapter: List[tuple[str, int]],
        enable_llm_expansion: bool,
        enable_confidence_calibration: bool,
        extraction_style: str # <-- NEW ARG
    ) -> Tuple[List[schemas.CEKEvent], List[schemas.EventProducesEntity], Dict[str, List[Tuple[str, int]]]]:
        """Process a batch of paragraphs in parallel (Standard Mode)."""
        all_events = []
        all_produces = []
        entity_occurrences_batch = defaultdict(list)
        
        try:
            results_batch = await llm_service.batch_extract_events(
                paragraphs_with_chapter, 
                self.openai_model,
                self.client,
                enable_llm_expansion,
                enable_confidence_calibration,
                extraction_style # Pass style
            )
            
            for (para_events_json, logprobs), (para, chapter_id) in zip(results_batch, paragraphs_with_chapter):
                events, produces, occ_batch = self._parse_event_json_data(
                    para_events_json, chapter_id, logprobs, enable_confidence_calibration
                )
                all_events.extend(events)
                all_produces.extend(produces)
                for key, val_list in occ_batch.items():
                    entity_occurrences_batch[key].extend(val_list)

        except Exception as e:
            print(f"[error] Batch processing failed: {e}")
        
        return all_events, all_produces, entity_occurrences_batch
    
    async def _process_text_chunk(
        self,
        text_chunk: str,
        chapter_id: int,
        enable_llm_expansion: bool,
        enable_confidence_calibration: bool,
        extraction_style: str # <-- NEW ARG
    ) -> Tuple[List[schemas.CEKEvent], List[schemas.EventProducesEntity], Dict[str, List[Tuple[str, int]]]]:
        """Process a single large chunk of text (Experimental Mode)."""
        try:
            events_json, logprobs = await llm_service.extract_events_from_text( # <-- Use the universal function
                text_chunk,
                chapter_id,
                self.openai_model,
                self.client,
                enable_llm_expansion,
                enable_confidence_calibration,
                extraction_style # Pass style
            )
            
            return self._parse_event_json_data(
                events_json, chapter_id, logprobs, enable_confidence_calibration
            )
        except Exception as e:
            print(f"[error] Text chunk processing failed: {e}")
            return [], [], defaultdict(list)

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
                # Use event names for pairing if quotes are empty (e.g. from chunking)
                cause_text = cause_ev.source_quote or cause_ev.name
                effect_text = effect_ev.source_quote or effect_ev.name
                pairs_to_assess.append((
                    cause_text, effect_text, cause_id, effect_id
                ))
        
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
                    rejected_edges += 1
                    continue
                
                cause_seq = self.dag_validator.event_sequence_map.get(cause_id)
                effect_seq = self.dag_validator.event_sequence_map.get(effect_id)
                
                if cause_seq is None or effect_seq is None:
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

    async def _batch_semantic_linking(
        self,
        events: List[schemas.CEKEvent],
        batch_size: int
    ) -> List[schemas.SemanticLink]:
        
        print("[semantic linking] Starting semantic link assessment...")
        semantic_links = []
        
        pairs_to_assess_set: Set[tuple[str, str]] = set()
        for i, ev in enumerate(events):
            start = max(0, i - 4) # Use same window
            for j in range(start, i):
                cand = events[j]
                if cand.sequence >= ev.sequence: continue
                pairs_to_assess_set.add((cand.id, ev.id))

        event_map_by_id = {ev.id: ev for ev in events}
        pairs_to_assess = []
        for cause_id, effect_id in pairs_to_assess_set:
            cause_ev = event_map_by_id.get(cause_id)
            effect_ev = event_map_by_id.get(effect_id)
            if cause_ev and effect_ev:
                pairs_to_assess.append((
                    cause_ev.source_quote or cause_ev.name,
                    effect_ev.source_quote or effect_ev.name,
                    cause_id,
                    effect_id
                ))
        
        print(f"[semantic linking] Assessing {len(pairs_to_assess)} pairs...")
        
        for batch_start in range(0, len(pairs_to_assess), batch_size):
            batch = pairs_to_assess[batch_start:batch_start + batch_size]
            
            assessments = await llm_service.batch_assess_semantic_pairs(
                batch, self.openai_model, self.client
            )
            
            for (cause_quote, effect_quote, cause_id, effect_id), assessment in zip(batch, assessments):
                if assessment is None or assessment.get("relation") == "none":
                    continue
                
                link = schemas.SemanticLink(
                    id=utils._make_id("sem"),
                    source_event_ids=[cause_id],
                    target_event_ids=[effect_id],
                    relation=assessment.get("relation", "explanation"),
                    cue=assessment.get("cue"),
                    confidence=float(assessment.get("confidence", 0.0))
                )
                semantic_links.append(link)

        print(f"[semantic linking] Found {len(semantic_links)} semantic links.")
        return semantic_links

    async def _generate_scenes(
        self,
        events: List[schemas.CEKEvent]
    ) -> List[schemas.Scene]:
        print("[scene grouping] Starting scene generation...")
        all_scenes = []
        
        events_by_chapter = defaultdict(list)
        for ev in events:
            events_by_chapter[ev.chapter].append(ev)
            
        for chapter_id, chapter_events in events_by_chapter.items():
            print(f"[scene grouping] Processing chapter {chapter_id}...")
            chapter_events.sort(key=lambda x: x.sequence)
            
            scene_data_list = await llm_service.extract_scenes_from_chapter_async(
                chapter_events, chapter_id, self.openai_model, self.client
            )
            
            for scene_data in scene_data_list:
                try:
                    scene = schemas.Scene(
                        id=utils._make_id("scene"),
                        event_ids=scene_data.get("event_ids", []),
                        chapter=chapter_id,
                        theme=scene_data.get("theme", "Untitled Scene"),
                        confidence=float(scene_data.get("confidence", 0.7))
                    )
                    all_scenes.append(scene)
                except Exception as e:
                    print(f"[warning] Failed to parse scene data: {e}")
        
        print(f"[scene grouping] Generated {len(all_scenes)} scenes.")
        return all_scenes

    # --- FIX IS HERE: Added 'extraction_style' to the function definition ---
    async def run_async(self, text_path: str, 
                       out_json: str,
                       out_cypher: str,
                       out_csv_dir: str,
                       max_chapters: Optional[int],
                       batch_size: int,
                       causal_window: int,
                       causal_sample_rate: float,
                       causal_batch_size: int,
                       # --- All new args are now included ---
                       paragraph_chunk_size: int,
                       extraction_style: str, # <--- THIS WAS THE MISSING ARGUMENT
                       graph_model: str,
                       enable_scene_grouping: bool,
                       enable_semantic_linking: bool,
                       enable_llm_expansion: bool,
                       enable_confidence_calibration: bool
                       ) -> Dict[str, Any]:
        """Main async pipeline orchestrator"""
        
        if enable_confidence_calibration and SENTENCE_TRANSFORMER_MODEL is None:
            print("[ERROR] '--enable-confidence-calibration' was passed, but 'sentence-transformers' is not installed.")
            print("Please run 'pip install sentence-transformers' and try again.")
            raise ImportError("'sentence-transformers' library not found.")
            
        print("[pipeline] Loading text...")
        raw = text_processor.load_text(text_path)
        chapters = text_processor.split_chapters(raw)

        if max_chapters is not None:
            chapters = chapters[:max_chapters]

        print(f"[pipeline] Processing {len(chapters)} chapters")

        all_events: List[schemas.CEKEvent] = []
        all_produces: List[schemas.EventProducesEntity] = []
        entity_occurrences = defaultdict(list)
        self.global_event_sequence = 0

        for chapter_id, chapter_text in chapters:
            paragraphs = text_processor.split_into_paragraphs(chapter_text)
            print(f"[chapter {chapter_id}] Processing {len(paragraphs)} paragraphs...")
            
            # --- LOGIC FOR CHUNKING VS PARALLEL ---
            
            # 1. CHUNKED MODE (chunk_size != 1)
            if paragraph_chunk_size != 1:
                # Handle "ALL" (0)
                if paragraph_chunk_size == 0:
                    chunk_size = len(paragraphs)
                else:
                    chunk_size = paragraph_chunk_size

                chunk_str = "ALL" if paragraph_chunk_size == 0 else str(chunk_size)
                print(f"[chapter {chapter_id}] Chunking: {chunk_str} paragraphs/call. Style: {extraction_style}")
                
                for i in range(0, len(paragraphs), chunk_size):
                    chunk = paragraphs[i : i + chunk_size]
                    text_chunk = "\n\n".join(chunk)
                    try:
                        events, produces, occ_batch = await self._process_text_chunk(
                            text_chunk, chapter_id,
                            enable_llm_expansion, enable_confidence_calibration,
                            extraction_style # <-- Pass style
                        )
                        all_events.extend(events)
                        all_produces.extend(produces)
                        for key, val_list in occ_batch.items():
                            entity_occurrences[key].extend(val_list)
                        print(f"[chapter {chapter_id}] Chunk {i//chunk_size + 1}: extracted {len(events)} events")
                    except Exception as e:
                        print(f"[error] Failed processing chunk {i//chunk_size + 1}: {e}")
            
            # 2. DEFAULT PARALLEL MODE (chunk_size == 1)
            else:
                for i in range(0, len(paragraphs), batch_size):
                    batch_paras = paragraphs[i:i+batch_size]
                    batch_with_chapter = [(para, chapter_id) for para in batch_paras]
                    try:
                        events, produces, occ_batch = await self._process_parallel_batch(
                            batch_with_chapter,
                            enable_llm_expansion, enable_confidence_calibration,
                            extraction_style # <-- Pass style
                        )
                        all_events.extend(events)
                        all_produces.extend(produces)
                        for key, val_list in occ_batch.items():
                            entity_occurrences[key].extend(val_list)
                        print(f"[chapter {chapter_id}] Batch {i//batch_size + 1}: extracted {len(events)} events")
                    except Exception as e:
                        print(f"[error] Failed processing batch {i//batch_size + 1}: {e}")

        if not all_events:
            raise schemas.CEKGError("No events were extracted")

        print(f"[pipeline] Total events extracted: {len(all_events)}")
        
        newly_produced, entity_occurrences = graph_builder.propagate_context(
            all_events, all_produces, entity_occurrences
        )
        all_produces.extend(newly_produced)
        print(f"[pipeline] Total event->entity productions after context: {len(all_produces)}")

        entity_to_event_links = graph_builder.create_entity_to_event_links(
            entity_occurrences, all_produces
        )

        # --- Causal Linking (Standard) ---
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
        
        # --- Semantic Linking (Experimental) ---
        semantic_links = []
        if enable_semantic_linking:
            try:
                semantic_links = await self._batch_semantic_linking(
                    events=all_events,
                    batch_size=causal_batch_size # Reuse batch size
                )
            except Exception as e:
                print(f"[warning] Semantic linking failed: {e}")
                traceback.print_exc()

        # --- Scene Grouping (Experimental) ---
        scenes = []
        if enable_scene_grouping:
            try:
                scenes = await self._generate_scenes(all_events)
            except Exception as e:
                print(f"[warning] Scene grouping failed: {e}")
                traceback.print_exc()

        # --- Export results ---
        print(f"[pipeline] Mapping data to generic graph (model: {graph_model})...")
        csvs = {}
        try:
            generic_nodes, generic_rels = graph_mapper.map_to_generic_graph(
                all_events, all_produces, entity_to_event_links, causal_links,
                graph_model=graph_model,
                semantic_links=semantic_links, # Pass new data
                scenes=scenes                 # Pass new data
            )
            
            print(f"[pipeline] Exporting results...")
            
            json_data = exporters.build_jsonld(
                all_events, all_produces, entity_to_event_links, causal_links
            )
            exporters.export_json(out_json, json_data)
            
            exporters.export_neo4j_cypher(
                out_cypher, generic_nodes, generic_rels
            )
            
            csvs = exporters.export_csv(
                out_csv_dir, all_events, all_produces, entity_to_event_links, causal_links,
                semantic_links=semantic_links,
                scenes=scenes
            )
        except Exception as e:
            print(f"[error] Export failed: {e}")
            traceback.print_exc()

        dag_stats = self.dag_validator.get_stats()
        cache_stats = await llm_service.get_cache_sizes()

        e2e_links = 0
        if graph_model == 'chain':
            e2e_links = len(entity_to_event_links)
        elif graph_model == 'star':
            e2e_links = len([p for p in all_produces if p.entity_type != 'whyfactor'])

        # --- Build final stats dict ---
        stats = {
            "events": len(all_events),
            "event_produces_entity": len(all_produces),
            "entity_points_to_event": e2e_links,
            "causal_links": len(causal_links),
            "cached_event_extractions": cache_stats["event_extraction_cache_size"],
            "cached_causal_assessments": cache_stats["assessment_cache_size"],
            "chapters_processed": len(set(ev.chapter for ev in all_events)),
            "dag_stats": dag_stats
        }
        if enable_semantic_linking:
            stats["semantic_links"] = len(semantic_links)
            stats["cached_semantic_assessments"] = cache_stats["semantic_cache_size"]
        if enable_scene_grouping:
            stats["scenes"] = len(scenes)
            stats["cached_scene_assessments"] = cache_stats["scene_cache_size"]

        return {
            "json": out_json, 
            "cypher": out_cypher, 
            "csv": csvs,
            "stats": stats
        }