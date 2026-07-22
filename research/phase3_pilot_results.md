# Phase 3 — Real-Data Pilot Results (evidence for the pivot)

**Question:** does a real long-term-memory dataset actually carry the redundancy our thesis assumes, and is it the kind (semantic, not lexical) that current methods miss? (This is the kill-criterion from [`novelty_analysis.md`](novelty_analysis.md) §9.)

**Data:** LoCoMo (`locomo10.json`, 10 multi-session conversations). We measure duplication in the dataset's own `observation` field — per-session, per-speaker atomic factual statements — the closest public analog to an agent's stored memory units. **2,541 facts total.**

**Method:** greedy single-link near-duplicate clustering; metrics `MDR = 1 − canonical/units` and `RTF = redundant-token fraction`. Tier 1 = lexical Jaccard (dependency-free); Tier 2 = semantic cosine with `all-MiniLM-L6-v2`. Scripts: [`../experiments/measure_duplication.py`](../experiments/measure_duplication.py), [`../experiments/measure_duplication_semantic.py`](../experiments/measure_duplication_semantic.py); raw output in `../experiments/*_output.txt`. Reproducible; no API, no fabrication.

## Results

| Method / threshold | MDR | RTF | Note |
|---|--:|--:|---|
| Lexical Jaccard (τ=0.6–1.0) | 0.000–0.009 | 0.000–0.007 | exact/lexical dedup finds essentially nothing |
| Semantic cosine τ=0.95 | 0.002 | 0.002 | verbatim restatements only |
| **Semantic cosine τ=0.85** | **0.10** | **0.10** | ~10% genuine paraphrase duplicates (per-conversation avg) |
| Semantic cosine τ=0.80 | 0.27 | 0.27 | over-merging starts |
| Semantic cosine τ=0.75 | 0.50 | 0.51 | merges related-but-distinct facts |

Verified example near-duplicate pairs (cosine ≥ 0.85), *within a single conversation's history*:
- 0.92 — "Caroline attended an LGBTQ+ pride parade last week…" ⇔ "Caroline attended a pride parade recently…"
- 0.87 — "Melanie values family moments and feels they make life awesome…" ⇔ "Melanie values family time and finds it special and important."

## Findings

1. **Real semantic duplication exists and is material (~10% at a conservative τ=0.85), and it is invisible to lexical/exact methods (~0%).** This is direct evidence for the *measurement* contribution: the redundancy axis is real and unmeasured by string-matching approaches. **The kill-criterion did not fire.**
2. **The τ=0.75–0.80 rows empirically reproduce the synthetic pilot's ρ effect:** past a point, a fixed similarity threshold merges *related-but-distinct* facts (e.g. "continuing education" vs "career in counseling"). This is concrete, real-data motivation for **query-aware canonicalization** over any fixed-threshold semantic dedup.

## Honesty / limitations
- One dataset; "memory units" = LoCoMo's own observation extractions, **not** the internal store of a deployed system (measuring Mem0/A-MEM/Zep stores needs running them with an LLM API — next).
- MDR/RTF are threshold-dependent; we report the whole curve, not one number.
- Semantic "duplicate" ≠ "safe to merge" — the ρ effect means the *right* metric pairs duplication with a task-utility check (the query-aware thesis).

## Effect on the go/no-go
- Kill-criterion (§9 of novelty_analysis): **cleared** — real semantic duplication is present.
- Measurement contribution: **now empirically grounded** on real data.
- Query-aware-canonicalization contribution: **motivated by real data**, not just the synthetic pilot.

---

## Phase 3b — Deployed-store duplication (real LLM extraction, local Ollama qwen3:8b)

**Upgrade of the claim:** instead of LoCoMo's own `observation` field, build the store the way Mem0/A-MEM/LangMem do — the local LLM extracts atomic facts per session, accumulated across the multi-session history — then measure duplication *of that store*. Two policies: **naive** (add every fact) vs **gated** (Mem0-style: skip if ≥0.85 cosine to an existing memory). Embedder: `nomic-embed-text`. Script: [`../experiments/build_and_measure_store.py`](../experiments/build_and_measure_store.py); output `../experiments/store_duplication_output.txt`. Local only, no API key. 6,834 facts extracted from 10 conversations (~5 h wall-clock; extractions + embeddings cached for instant re-runs).

| policy (avg store size) | τ=0.95 | τ=0.90 | τ=0.85 | τ=0.80 | τ=0.75 |
|---|--:|--:|--:|--:|--:|
| naive (683 facts/conv) — MDR | 0.06 | 0.29 | **0.62** | 0.87 | 0.96 |
| gated@0.85 (384 facts/conv) — MDR | 0.00 | 0.00 | 0.00 | **0.64** | 0.92 |

**Findings:**
1. A naive extract-then-store pipeline accumulates **large semantic duplication** (MDR 0.62 at τ=0.85) — real, measured.
2. The Mem0-style gate removes ~44% of facts on ingest (683→384) and zeroes duplication at its own threshold — standard dedup works *at its threshold*.
3. **A large near-duplicate band survives just below the gate** (MDR 0.64 at τ=0.80 even after gating at 0.85), and lowering the gate to catch it enters the ρ over-merge regime. **A single fixed similarity threshold cannot separate "duplicate" from "distinct."** This is the concrete gap query-aware canonicalization addresses — now demonstrated against a real system's own dedup mechanism.

**Caveats (important):**
- The naive MDR is **inflated by qwen3's extraction verbosity** (6,834 facts vs LoCoMo's 2,541 observations); a better extractor lowers the baseline. The robust, model-independent finding is the **relative** result (gate leaves a near-dup band; can't lower it without over-merging), not the absolute naive numbers.
- Thresholds are **embedder-specific**: this store used `nomic-embed`, the observation-stream run used `MiniLM` — do not compare absolute numbers across models.
- `MDR=0` at τ≥gate is **by construction**, not a discovery.
- Single dataset, single extractor + embedder.

**Effect on the thesis:** the measurement contribution and the query-aware-canonicalization motivation now hold on three independent angles that agree — (a) synthetic pilot, (b) LoCoMo observation stream, (c) a real LLM-built store with a real dedup gate. Remaining for publication-grade evidence: a better/standard extractor (Mem0/A-MEM) to pin the absolute baseline, plus a second dataset (LongMemEval) for generalization.

---

## Phase 3c — Does dedup affect ANSWER QUALITY? (the "without degrading task performance" half)

Three memory policies answer LoCoMo QA (cats 1–4) via RAG over the per-conversation fact store (qwen3:8b answerer, nomic-embed retrieval, k=8 facts/question, n=200). Script: [`../experiments/qa_dedup_eval.py`](../experiments/qa_dedup_eval.py); output `../experiments/qa_dedup_output.txt`.
- **full** — top-k from the full store (no dedup)
- **naive_dedup** — store blindly canonicalized at τ=0.82 (≈76% smaller), then top-k
- **query_aware** — full store, top-k by MMR (relevance − redundancy) [classic composer, proxy for the query-aware idea]

| policy | F1 | containment | ctx tokens/q |
|---|--:|--:|--:|
| full | 0.260 | 0.160 | 77.5 |
| **naive_dedup** | **0.146** | **0.085** | 77.7 |
| **query_aware** | **0.270** | **0.165** | 79.0 |

**Findings:**
1. **Naive/blind semantic dedup severely degrades QA: F1 −44%, containment −47%.** Real-data confirmation of the over-merge (ρ) risk — you cannot compact a memory store by blindly merging near-duplicates.
2. **Query-aware composition preserves quality** (matches full: 0.270 vs 0.260). Redundancy-aware selection is *safe*.

**What is NOT shown (honest):**
- **Token/storage efficiency was not demonstrated.** k=8 was fixed, so context tokens are ~equal across arms (77–79) by construction; query_aware preserved quality but did not *save* tokens. This tested accuracy-at-fixed-budget, not tokens-at-fixed-accuracy.
- **Absolute accuracy is low** (F1 0.26 / containment 0.16): the qwen3-extraction+retrieval pipeline is weak in absolute terms (drops specifics such as dates). Only the *relative* cross-arm comparison (same pipeline) is valid; absolute numbers are not competitive — flagged, not hidden.

**Net:** the *necessity* of query-awareness is strongly supported (naive dedup breaks QA; query-aware is safe). The *efficiency* claim (same quality at fewer tokens/facts) still needs a **budget sweep** (vary k / store-compaction and plot accuracy vs tokens) — the next experiment.
