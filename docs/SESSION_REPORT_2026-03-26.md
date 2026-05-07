# CEKG Session Report — 2026-03-26

> This report combines the pipeline run metrics, code changes, known issues, and improvement areas from the 2026-03-26 development session.

---

## 1. Pipeline Run Summary

| Metric | This Run | Previous Baseline | Change |
|--------|----------|-------------------|--------|
| Input | Great Expectations.txt (59 chapters) | — | — |
| Mode | `--full --max-pairs 999999 --resume` | — | — |
| Runtime | ~512 minutes (8h 32m) | — | — |
| Model | gpt-4o-mini | gpt-4o-mini | — |
| Events extracted | 3,338 | ~3,338 | — |
| Characters | 116 | — | — |
| Candidate pairs evaluated | **23,203** | 7,272 | **+219%** |
| Causal links total | **4,121** | 3,564 | **+15.6%** |
| — McKee | 1,825 | — | — |
| — Truby | 2,296 | — | — |
| Scenes | 295 | — | — |
| DAG valid | ✓ | ✓ | — |
| Orphan events | 0 | 604 | **fixed** |

Graph export: **4,861 nodes**, **22,399 relationships**, **27,260 Cypher statements**.

---

## 2. Code Changes (This Session)

### 2a. `cekg_pipeline/theme_annotation.py` — Expanded Supertype Mapping

**What changed:** The `FINE_TO_SUPERTYPE` dictionary was significantly expanded from ~30 entries to ~90 entries. Two new supertypes were introduced: `EMOTIONAL_DRIVE` and `SOCIAL_BOND`. Existing supertypes (`CAUSAL_PRODUCTION`, `CAUSAL_CONSTRAINT`, `NARRATIVE_ESCALATION`, `NARRATIVE_RESOLUTION`, `REVELATION_EPISTEMIC`, `MEDIATION_TRANSFER`, `THEMATIC_CONTRAST`, `THEMATIC_EXPLANATION`) received many additional fine-grained relation types that the LLM produces in practice but were previously unmapped (and therefore left as empty strings in the output).

**New supertypes added:**

| Supertype | Example fine-grained types |
|-----------|---------------------------|
| `EMOTIONAL_DRIVE` | `EMOTIONAL_TRIGGER`, `COMPASSION_TRIGGER`, `PSYCHOLOGICAL_IMPACT`, `CRUELTY_PLEASURE`, `EMOTIONAL_MANIPULATION` |
| `SOCIAL_BOND` | `ALLY_DEPENDENCE`, `FAMILY_INFLUENCE`, `MENTORSHIP_SUPPORT`, `PERSUASION_ATTEMPT`, `MOTIVATES` |

**Why:** After the full Great Expectations run, many causal links had an empty `edge_supertype` because their `relation_type` (e.g. `EMOTIONAL_TRIGGER`, `ALLY_SUPPORT`) was not in the mapping. This expansion ensures near-complete supertype coverage across the observed relation vocabulary.

---

### 2b. `cekg_pipeline/graph_mapper.py` — Export `edge_supertype` as Edge Property

**What changed:** `CausalLink` edges now include `edge_supertype` as an exported property on graph relationships (alongside `weight`, `confidence`, `theory`, `directionality`).

**Why:** `edge_supertype` was computed and stored on `CausalLink` objects but was silently dropped at export time, making it unavailable in Neo4j and the JSON-LD output. Adding it here makes thematic grouping queryable directly from the graph.

---

### 2c. `cekg_pipeline/exporters.py` — Fix Cypher MERGE to Always Update Properties

**What changed:** In `export_neo4j_cypher`, the relationship MERGE pattern changed from:
```cypher
MERGE (a)-[r:TYPE]->(b) ON CREATE SET r = {props}
```
to:
```cypher
MERGE (a)-[r:TYPE]->(b) SET r = {props}
```

**Why:** `ON CREATE SET` only applied properties when the relationship was first created. On re-import (e.g. after a `--resume` run or schema change), existing relationships retained stale properties. `SET r = {props}` unconditionally refreshes properties on every import, which is the correct behavior for an idempotent import script.

---

### 2d. `cekg_pipeline/pipeline.py` — Re-apply Supertypes at Export Time

**What changed:** `_export_results` now calls `assign_edge_supertypes(causal_links)` immediately before the export block.

**Why:** When resuming from a linking checkpoint (`--resume`), `CausalLink` objects are deserialized from pickle. If `FINE_TO_SUPERTYPE` was updated since the checkpoint was saved (as happened this session), the deserialized links have stale or missing `edge_supertype` values. Re-applying the mapping at export time ensures that every export — whether fresh or resumed — reflects the current mapping.

---

## 3. New Feature Validation (From This Run)

### SAT Sentence Segmentation (`wtpsplit`)
- **Status:** Active and working
- Replaces naive `split('. ')` for chunk boundaries during event extraction
- No measurable regression in event count; cleaner chunk boundaries reduce mid-sentence cuts

### BM25 Keyword Candidate Pairs
- **Status:** Active — largest contributor to pair expansion

| Pool | Unique Pairs | Share |
|------|-------------|-------|
| Adjacent (i, i+1) | 3,337 | 14% |
| Entity-guided | 2,696 | 12% |
| **BM25 keyword** | **11,304** | **49%** |
| Scene (sim ≥ 0.5) | 3,132 | 13% |
| Long-shot sliding | 2,734 | 12% |
| **Total** | **23,203** | |

BM25 is the dominant pool, contributing nearly half of all candidate pairs. Named entities and domain terms that embedding similarity compresses away (e.g. "Magwitch", "convict", "debt") produce high BM25 scores across the full span of the novel.

### RAG Passage Context for Causal Assessment
- **Status:** Active
- Index: **2,879 passages** (400-char segments, 80-char overlap)
- Top-2 passages injected per pair into the causal LLM prompt
- Provides grounding for ambiguous long-range pairs; effect on link quality not directly measurable in one run

### Stale CSV Cleanup
- **Status:** Fixed — `thematic_links.csv` (82,681 rows, old format) removed automatically
- Current output: 10 active CSV files, 31,820 total rows

### Orphan Scene Fallback
- **Status:** Fixed — 0 orphan events (down from 604)
- 295 scenes; catch-all fallback active for chapters where the LLM scene grouper missed events

---

## 4. Output File Summary

| File | Rows | Description |
|------|------|-------------|
| `events.csv` | 3,338 | Event nodes with `theme_annotations`, `edge_supertype` |
| `causes.csv` | 4,121 | CAUSES edges (McKee + Truby) |
| `follows.csv` | 3,279 | FOLLOWS (chronological) edges |
| `agents.csv` | 5,073 | Character/agent nodes |
| `acts_in.csv` | 4,190 | ACTS_IN edges |
| `affected_in.csv` | 883 | AFFECTED_IN edges |
| `motivates.csv` | 3,629 | MOTIVATES edges |
| `whyfactors.csv` | 3,629 | WHY_FACTOR nodes |
| `scenes.csv` | 295 | Scene nodes |
| `scene_includes_event.csv` | 3,373 | SCENE_INCLUDES_EVENT edges |
| `hosts.csv` | 0 | Empty — `place_type` not populated |
| `places.csv` | 0 | Empty — place nodes not generated |

---

## 5. Known Issues

### Chapter 59: 7/10 chunks failed (70% data loss)

**Root cause:** `_async_llm_json_call` internally caught all exceptions and retried with 1–2s backoff, returning `[]` silently. This meant `_process_chunk_with_retry`'s rate-limit-aware 30–90s retry logic never activated — it never saw an exception.

By the time chapter 59 runs (last in the 8h session), the per-minute RPM quota is depleted from chapter 58's burst. The 3 internal short retries exhaust the remaining tokens, and 7 of 10 chunks return `[]`.

**Fix applied (this session):** `_async_llm_json_call` now re-raises immediately on rate-limit errors (HTTP 429, "rate limit", "quota"), allowing `_process_chunk_with_retry`'s 5-retry loop with 30–90s progressive backoff to activate.

**Residual impact:** Chapter 59 contributed only ~26 events instead of the expected ~85–90. The epilogue (Pip's final reunion with Estella, closure of the Magwitch arc) may be underrepresented in the knowledge graph.

---

## 6. Areas for Improvement

### High Priority

**1. Scene generation is sequential (slow)**
`_generate_scenes_optimized` iterates chapters in a `for` loop — 59 blocking API calls in series. Total scene generation time: estimated 20–30 minutes.
*Fix:* `asyncio.gather(...)` with a concurrency semaphore.

**2. Scene extraction token truncation**
Chapter 1's scene response was 33,271 chars and hit the 12,000-token cap. JSON parse failed; fallback created a single catch-all scene for the entire chapter.
*Fix:* Increase `max_tokens` to 16,000 for scene extraction, or split large chapters before sending.

**3. Events with no causal bridge**
Some events are isolated (no incoming or outgoing CAUSES edge). Unique or generic entity descriptions may not appear in any candidate pair pool.
*Potential fix:* Add a "near-miss" pool: events within ±5 positions that share no entities but are not in adjacent pairs.

**4. Chapter 59 data loss**
~26 events instead of ~85–90. The rate-limit fix is in place but the existing data cannot be recovered without a full re-run or targeted partial re-run of chapter 59.

### Medium Priority

**5. BM25 top-K=5 may miss long-range pairs**
For characters appearing in 20+ events (Pip, Magwitch, Estella, Miss Havisham), the top-5 BM25 results cluster near high-frequency local events.
*Fix:* Increase `BM25_TOP_K` to 10–15.

**6. RAG contribution not measurable**
No logging of whether RAG context changes `relationType` decisions.
*Fix:* A/B test a subset of pairs with and without RAG context to quantify the delta.

**7. Pair generation not cached**
BM25/similarity pair generation recomputes every run (~30s for 3,338 events). `--resume` does not skip this step.
*Fix:* Cache the pair list as part of the linking checkpoint.

### Low Priority

**8. Catch-all scenes conflate unrelated events**
In later chapters with dense action, a single catch-all scene may contain 30+ unrelated events, inflating within-scene pair counts.

**9. `hosts.csv` and `places.csv` are empty**
`place_type` is never set on Scene objects. If places are needed in Neo4j, the scene exporter should resolve location strings to canonical place nodes.

**10. Cypher import script references `semantic_links`**
The old `semantic_links.csv` (merged CAUSES + FOLLOWS) is gone. Downstream Neo4j queries still referencing `semantic_links` will need updating.

---

---

# CEKG 세션 리포트 — 2026년 3월 26일

> 이 리포트는 2026-03-26 개발 세션의 파이프라인 실행 메트릭, 코드 변경사항, 알려진 이슈, 개선 영역을 통합하여 정리한 문서입니다.

---

## 1. 파이프라인 실행 요약

| 항목 | 이번 실행 | 이전 기준선 | 변화 |
|------|----------|------------|------|
| 입력 | Great Expectations.txt (59개 챕터) | — | — |
| 모드 | `--full --max-pairs 999999 --resume` | — | — |
| 실행 시간 | ~512분 (8시간 32분) | — | — |
| 모델 | gpt-4o-mini | gpt-4o-mini | — |
| 추출 이벤트 수 | 3,338 | ~3,338 | — |
| 등장인물 수 | 116 | — | — |
| 평가된 후보 쌍 수 | **23,203** | 7,272 | **+219%** |
| 인과 링크 총 수 | **4,121** | 3,564 | **+15.6%** |
| — McKee | 1,825 | — | — |
| — Truby | 2,296 | — | — |
| 장면 수 | 295 | — | — |
| DAG 유효성 | ✓ | ✓ | — |
| 고아 이벤트 수 | 0 | 604 | **수정됨** |

그래프 내보내기 결과: **노드 4,861개**, **관계 22,399개**, **Cypher 구문 27,260개**.

---

## 2. 코드 변경 사항 (이번 세션)

### 2a. `cekg_pipeline/theme_annotation.py` — 슈퍼타입 매핑 확장

**변경 내용:** `FINE_TO_SUPERTYPE` 딕셔너리가 약 30개 항목에서 약 90개 항목으로 대폭 확장되었습니다. 두 가지 새 슈퍼타입 `EMOTIONAL_DRIVE`(감정적 동인)와 `SOCIAL_BOND`(사회적 유대)가 추가되었습니다. 기존 슈퍼타입들도 LLM이 실제로 생성하지만 이전에는 매핑되지 않아 출력에서 빈 문자열로 남겨졌던 세부 관계 타입들을 포함하도록 업데이트되었습니다.

**새로 추가된 슈퍼타입:**

| 슈퍼타입 | 대표 세부 타입 예시 |
|---------|-----------------|
| `EMOTIONAL_DRIVE` | `EMOTIONAL_TRIGGER`, `COMPASSION_TRIGGER`, `PSYCHOLOGICAL_IMPACT`, `CRUELTY_PLEASURE` |
| `SOCIAL_BOND` | `ALLY_DEPENDENCE`, `FAMILY_INFLUENCE`, `MENTORSHIP_SUPPORT`, `PERSUASION_ATTEMPT` |

**이유:** Great Expectations 전체 실행 후 많은 인과 링크에 `edge_supertype`이 비어 있었는데, 이는 해당 `relation_type`(예: `EMOTIONAL_TRIGGER`, `ALLY_SUPPORT`)이 매핑 딕셔너리에 없었기 때문입니다. 이번 확장으로 관찰된 관계 어휘 전반에 걸쳐 거의 완전한 슈퍼타입 커버리지를 확보합니다.

---

### 2b. `cekg_pipeline/graph_mapper.py` — `edge_supertype` 엣지 속성 내보내기

**변경 내용:** `CausalLink` 엣지에 `edge_supertype`이 그래프 관계의 속성으로 내보내지도록 추가되었습니다(`weight`, `confidence`, `theory`, `directionality`와 함께).

**이유:** `edge_supertype`은 `CausalLink` 객체에 계산·저장되고 있었지만, 내보내기 시점에 조용히 누락되어 Neo4j 및 JSON-LD 출력에서 사용할 수 없었습니다. 이를 추가함으로써 주제별 그룹화가 그래프에서 직접 쿼리 가능해집니다.

---

### 2c. `cekg_pipeline/exporters.py` — Cypher MERGE가 항상 속성을 업데이트하도록 수정

**변경 내용:** `export_neo4j_cypher`에서 관계 MERGE 패턴이 다음과 같이 변경되었습니다:
```cypher
-- 이전
MERGE (a)-[r:TYPE]->(b) ON CREATE SET r = {props}

-- 이후
MERGE (a)-[r:TYPE]->(b) SET r = {props}
```

**이유:** `ON CREATE SET`은 관계가 처음 생성될 때만 속성을 적용합니다. 재임포트 시(예: `--resume` 실행 후 또는 스키마 변경 후) 기존 관계는 이전 속성을 그대로 유지했습니다. `SET r = {props}`는 모든 임포트에서 속성을 무조건 갱신하므로, 멱등적 임포트 스크립트의 올바른 동작 방식입니다.

---

### 2d. `cekg_pipeline/pipeline.py` — 내보내기 시점에 슈퍼타입 재적용

**변경 내용:** `_export_results`가 내보내기 블록 직전에 `assign_edge_supertypes(causal_links)`를 호출합니다.

**이유:** 링킹 체크포인트에서 재개(`--resume`)할 때, `CausalLink` 객체가 피클에서 역직렬화됩니다. 이번 세션처럼 체크포인트 저장 이후에 `FINE_TO_SUPERTYPE`이 업데이트된 경우, 역직렬화된 링크는 낡거나 누락된 `edge_supertype` 값을 가집니다. 내보내기 시점에 매핑을 재적용함으로써 신규 실행이든 재개 실행이든 모든 내보내기가 현재 매핑을 반영합니다.

---

## 3. 신규 기능 검증 (이번 실행 결과)

### SAT 문장 분절 (`wtpsplit`)
- **상태:** 활성화, 정상 작동
- 청크 분할 시 단순 `split('. ')` 대신 적절한 문장 경계를 감지
- 이벤트 수에 측정 가능한 회귀 없음; 문장 중간 절단 현상 감소

### BM25 키워드 후보 쌍
- **상태:** 활성화 — 쌍 확장의 최대 기여 풀

| 풀 | 고유 쌍 수 | 비율 |
|----|-----------|------|
| 인접 (i, i+1) | 3,337 | 14% |
| 엔티티 기반 | 2,696 | 12% |
| **BM25 키워드** | **11,304** | **49%** |
| 장면 (sim ≥ 0.5) | 3,132 | 13% |
| 장거리 슬라이딩 | 2,734 | 12% |
| **합계** | **23,203** | |

BM25는 전체 후보 쌍의 절반 가까이를 기여하는 지배적인 풀입니다. 임베딩 유사도가 압축해버리는 고유명사 및 도메인 용어(예: "Magwitch", "convict", "debt")가 소설 전체에 걸쳐 높은 BM25 점수를 생성합니다.

### RAG 구절 컨텍스트 (인과 평가용)
- **상태:** 활성화
- 인덱스: **2,879개 구절** (400자 세그먼트, 80자 오버랩)
- 인과 LLM 프롬프트마다 상위 2개 구절 삽입
- 모호한 장거리 쌍에 근거를 제공; 1회 실행으로는 링크 품질 영향 측정 불가

### 오래된 CSV 자동 삭제
- **상태:** 수정됨 — `thematic_links.csv`(82,681행, 이전 형식) 자동 삭제
- 현재 출력: 활성 CSV 파일 10개, 총 31,820행

### 고아 장면 폴백
- **상태:** 수정됨 — 고아 이벤트 0개 (604개에서 감소)
- 총 295개 장면; LLM 장면 그루퍼가 이벤트를 놓친 챕터에 catch-all 폴백 적용

---

## 4. 출력 파일 요약

| 파일 | 행 수 | 설명 |
|------|------|------|
| `events.csv` | 3,338 | `theme_annotations`, `edge_supertype` 포함 이벤트 노드 |
| `causes.csv` | 4,121 | CAUSES 엣지 (McKee + Truby) |
| `follows.csv` | 3,279 | FOLLOWS (시간순) 엣지 |
| `agents.csv` | 5,073 | 등장인물/에이전트 노드 |
| `acts_in.csv` | 4,190 | ACTS_IN 엣지 |
| `affected_in.csv` | 883 | AFFECTED_IN 엣지 |
| `motivates.csv` | 3,629 | MOTIVATES 엣지 |
| `whyfactors.csv` | 3,629 | WHY_FACTOR 노드 |
| `scenes.csv` | 295 | 장면 노드 |
| `scene_includes_event.csv` | 3,373 | SCENE_INCLUDES_EVENT 엣지 |
| `hosts.csv` | 0 | 비어 있음 — `place_type` 미설정 |
| `places.csv` | 0 | 비어 있음 — 장소 노드 미생성 |

---

## 5. 알려진 이슈

### 챕터 59: 청크 10개 중 7개 실패 (70% 데이터 손실)

**근본 원인:** `_async_llm_json_call`이 내부적으로 모든 예외를 잡아 1–2초 백오프로 재시도한 후 조용히 `[]`를 반환했습니다. 이로 인해 `_process_chunk_with_retry`의 레이트 리밋 인식 30–90초 재시도 로직이 한 번도 활성화되지 못했습니다.

챕터 59는 8시간 세션의 마지막 챕터로, 챕터 58의 버스트 처리로 분당 RPM 할당이 소진된 상태였습니다.

**적용된 수정:** `_async_llm_json_call`이 레이트 리밋 오류(HTTP 429, "rate limit", "quota")에서 즉시 예외를 재발생시키도록 수정. `_process_chunk_with_retry`의 5회 재시도 루프(30–90초 프로그레시브 백오프)가 정상 작동합니다.

**잔여 영향:** 챕터 59에서 예상 ~85–90개 대신 약 26개의 이벤트만 추출됨. 소설 에필로그(Pip의 Estella와의 최종 재회, Magwitch 아크의 마무리)가 지식 그래프에서 과소 표현될 수 있습니다.

---

## 6. 개선 영역

### 높은 우선순위

**1. 장면 생성이 순차 처리 (느림)**
`_generate_scenes_optimized`가 챕터를 `for` 루프로 순차 처리 — 59개의 블로킹 API 호출이 직렬로 실행됩니다.
*수정 방안:* `asyncio.gather(...)`와 동시성 세마포어 적용.

**2. 장면 추출 토큰 잘림**
챕터 1의 장면 응답이 33,271자로 12,000 토큰 상한에 도달. JSON 파싱 실패; 폴백으로 챕터 전체에 대한 단일 catch-all 장면 생성.
*수정 방안:* 장면 추출의 `max_tokens`를 16,000으로 높이거나, 대형 챕터를 전송 전에 분할.

**3. 인과 연결이 없는 이벤트 존재**
일부 이벤트에 인입/인출 CAUSES 엣지가 없음. 고유하거나 일반적인 엔티티 설명이 어떤 후보 쌍 풀에도 포함되지 않을 수 있습니다.
*잠재적 개선:* "근접 미스" 풀 추가: 엔티티를 공유하지 않지만 인접 쌍에도 없는 ±5 위치 이내 이벤트.

**4. 챕터 59 데이터 손실**
레이트 리밋 수정은 적용되었으나 기존 데이터는 전체 재실행 또는 챕터 59 부분 재실행 없이는 복구 불가.

### 중간 우선순위

**5. BM25 상위-K=5가 장거리 쌍을 놓칠 수 있음**
20회 이상 등장하는 인물(Pip, Magwitch, Estella, Miss Havisham)의 경우 상위 5개 BM25 결과가 빈도 높은 근처 이벤트로 몰릴 수 있습니다.
*수정 방안:* `BM25_TOP_K`를 10–15로 높임.

**6. RAG 기여도 측정 불가**
RAG 컨텍스트가 `relationType` 결정을 변경하는지 로깅 없음.
*수정 방안:* 일부 쌍에 대해 RAG 유무 A/B 테스트 실시.

**7. 쌍 생성 결과가 캐시되지 않음**
BM25/유사도 쌍 생성이 매 실행마다 재계산됨(3,338개 이벤트 기준 약 30초). `--resume` 시에도 이 단계는 건너뛰지 않습니다.
*수정 방안:* 쌍 목록을 링킹 체크포인트의 일부로 캐시.

### 낮은 우선순위

**8. Catch-all 장면이 관련 없는 이벤트를 묶음**
후반부 챕터에서 단일 catch-all 장면이 30개 이상의 무관한 이벤트를 포함할 수 있어, `dynamic_context`의 장면 풀에서 장면 내 쌍 수가 과도하게 늘어납니다.

**9. `hosts.csv`와 `places.csv`가 비어 있음**
Scene 객체에 `place_type`이 설정되지 않음. Neo4j에서 장소가 필요하다면, 장면 내보내기가 위치 문자열을 정규 장소 노드로 해석해야 합니다.

**10. Cypher 임포트 스크립트가 `semantic_links` 참조**
기존 `semantic_links.csv`(CAUSES + FOLLOWS 합산)는 삭제되었습니다. `semantic_links`를 참조하는 다운스트림 Neo4j 쿼리 업데이트가 필요합니다.
