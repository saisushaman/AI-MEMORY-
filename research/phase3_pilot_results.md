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

**Next real experiment (to make it publishable):** run one deployed system (Mem0 or A-MEM) on LoCoMo, snapshot its actual stored memories, and measure MDR/RTF *of that store* — turning "the extracted-observation stream is redundant" into "system X stores N% duplicate memories." Requires an LLM API key (extraction). Flag before running.
