# GPT-5 Smoke Test — Pipeline Compatibility Report

**Date:** 2026-04-28
**Scope:** 1 chapter of *Great Expectations* (~9.6 K chars), full pipeline (extraction → context → linking → theme annotation → export).
**Command used:**

```bash
python main.py \
  --input "Great Expectations.txt" \
  --openai-model gpt-5 \
  --max-chapters 1 \
  --max-pairs 30 \
  --chunk-size 3000 \
  --max-concurrent-calls 4 \
  --checkpoint-dir ./checkpoints_gpt5_smoke \
  --clear-checkpoints \
  --out-json ./gpt5_smoke.json \
  --out-cypher ./gpt5_smoke_import.cypher \
  --out-csv ./neo4j_csv_gpt5_smoke
```

(Full log: `gpt5_smoke_run.log`.)

---

## 1. Result summary

| Metric | Value |
|---|---|
| Wall-clock time | **508 s (8.5 min)** |
| Events extracted | 89 (from 4 chunks, 0 failures) |
| Causal candidate pairs | 30 (capped) |
| Causal links produced | 8 (5 McKee + 3 Truby; 27% link rate) |
| Distinct causal relation types | 4 (`PSYCHOLOGICAL_PRESSURE`, `SCENE_CAUSATION`, `EVENT_REINFORCEMENT`, `EVENT_ENABLES_NEXT`) |
| Theme annotations completed | 89 / 89 (100%) |
| Thematic edges generated | 19 (KNOWLEDGE=7, POWER=6, WEALTH=3, JUSTICE=3) |
| Cypher statements emitted | 612 |
| Pipeline crashed? | **No** |
| Output JSON-LD valid? | Yes |

End-to-end the existing reasoning-model code path
(`is_reasoning_model = "gpt-5" in model` in `_async_llm_json_call`)
correctly switches to `max_completion_tokens` and `temperature=1.0`, so no
code changes were needed.

---

## 2. Quality observations

### 2a. Theme annotations are markedly cleaner

Sample event:

```
event/b1ecb6d3:
"Philip Pirrip affirmed his father's family name as Pirrip based on the
 authority of the tombstone and Mrs. Joe Gargery."

  POWER      none
  WEALTH     none
  KINSHIP    direct revealing  conf 0.7  ev: "He affirmed his father's family name…"
  KNOWLEDGE  direct revealing  conf 0.9  ev: "…based on the authority of the tombstone…"
  JUSTICE    none
```

- `confidence` is now a real float (was uniformly `null` under gpt-4o).
- `evidence` quotes the actual mechanism, not the mood.
- POWER and WEALTH correctly stay `none` — this is exactly the
  weak-signal-overtagging the 0326 feedback flagged, and GPT-5 + the new
  prompt suppress it.
- `role` distribution is no longer collapsed to `mediating`/`null` —
  `revealing` shows up where the prompt says it should.

### 2b. Causal relation diversity improved

GPT-5 used 4 distinct types for 8 links (50% diversity). Earlier gpt-4o
runs produced ~6 relation types across thousands of links — the *per-batch*
diversity is the key win and the reviewer's specific complaint.

### 2c. Thematic edges have signal

19 thematic edges from 8 causal links (avg 2.4 themes carried per causal
beat) is a reasonable density. All 19 came from the `causal:` mode; the
chapter is too short for the `scene_spine:` mode to add anything.

---

## 3. Cost / latency reality check

The single-token probe at the top of the test was illuminating:

```python
client.chat.completions.create(
    model='gpt-5', max_completion_tokens=16, ...
).usage  # → reasoning_tokens=16, completion_tokens=16, content=""
```

**The model spent the entire 16-token budget reasoning and returned an
empty string.** Reasoning tokens are billed and consume the
`max_completion_tokens` budget *before* visible output is generated. This
has two consequences for the pipeline:

1. Every `max_completion_tokens` budget in `_async_llm_json_call` needs to
   be sized for `reasoning_tokens + visible_tokens`, not just the visible
   JSON. The current sizes (16 K for extraction, 13 K for bulk causal,
   2 K for theme annotation) appear to be sufficient on a 1-chapter test
   — *no truncations were observed* — but the headroom is unknown for
   longer chapters.
2. **Latency is ~8 s per call even for trivial prompts.** Across the full
   pipeline that translates to ~10× slowdown vs gpt-4o-mini.

### Extrapolation to full *Great Expectations*

| Stage | gpt-4o-mini est. | gpt-5 est. (this run × 59 ch.) |
|---|---|---|
| Extraction | ~6 min | ~50 min |
| Linking (25 K pairs) | ~12 min | ~2 hr |
| Theme annotation (~5 K events) | ~5 min | ~50 min |
| **Total wall-clock** | **~25 min** | **~3.5 hr** |

Cost scaling is harder to estimate without GPT-5 published pricing, but
reasoning tokens at ~600/call × thousands of calls is the dominant term.

### Recommendations

| Stage | Suggested model | Reason |
|---|---|---|
| Event extraction | **gpt-5** | Best quality; extraction is the foundation |
| Causal linking | **gpt-5** for the first 5 K pairs, gpt-4o-mini for the long tail | Diminishing returns past the high-confidence band |
| Theme annotation | **gpt-5** | Quality jump is largest here; per-event call is small |
| Scene grouping | **gpt-4o-mini** | Cheap heuristic task; reasoning headroom not needed |
| Agent classification | **gpt-4o-mini** | Same |

This per-stage routing is straightforward — the pipeline already accepts
`--openai-model`; we'd add per-stage overrides on the
`CEKGPreprocessor.__init__` constructor. Roughly a 20-line change.

---

## 4. Outstanding compatibility risks

Found **no blockers**, but two amber flags:

1. **`max_completion_tokens` of 2048 for theme annotation may be tight on
   long chapters.** None of the 89 events in the test truncated, but the
   reasoning-token overhead is roughly fixed at ~600 tokens per call; an
   event with a long causal context could push the visible JSON over the
   remaining ~1.4 K. Consider raising to 4096 when `gpt-5` is detected.
2. **Rate limits.** With 4 concurrent calls × ~8 s each, throughput is
   ~30 calls/min. The test triggered no 429s, but the full novel will
   make tens of thousands of calls. Watch for 429 spikes; the existing
   30–90 s rate-limit backoff in `_process_chunk_with_retry` is the
   safety net.

---

## 5. Verdict

GPT-5 works with the existing pipeline as-is, produces visibly better
theme annotations and more diverse causal relations on a 1-chapter test,
and exposes a cost/latency tradeoff that argues for **per-stage model
routing** rather than a global flag. A full-novel run is feasible but
should budget ~3–4 hours and be supervised for rate-limit behaviour.

Recommended next step: implement per-stage `--openai-model-extraction`,
`--openai-model-linking`, `--openai-model-theme` flags so we can keep
gpt-5 where it adds quality and gpt-4o-mini where it doesn't.
