# Direction 3 (Memory Faithfulness) — Verdict: dead two ways (2026-07-30)

Novelty gate (3 verified surveys) + a phenomenon-check pilot, same discipline as before.

## 1. Pre-empted (novelty gate failed)
The exact idea — hallucinated memories accumulating and propagating across sessions — is already an active 2025-26 area (all arXiv-verified):
- **HaluMem** (2511.03506, Nov 2025) — *first operation-level hallucination benchmark for agent memory*; shows memory systems "generate and accumulate hallucinations during extraction and updating" and tracks how they "arise, propagate, and impact final outputs." (GitHub + OpenReview corroborated.)
- **ConsistencyGate** (2607.22962) — coins "memory contamination": *a hallucinated fact written at one step persists as a false premise for every subsequent step*; write-time self-consistency gate.
- **MemGuard** (2605.28009), **TrustMem** (2606.25161) — two more 2026 mitigations for hallucinated content entering memory.
- Framing/taxonomy already exists (survey 2604.16548, "memory hallucination = internal-origin corruption").
- The measurement methodology is a fully mature, solved area: FActScore, SAFE, RefChecker, MiniCheck, SummaC, QAFactEval, WiCE, GraphJudge. FActScore already reports ~42% unsupported atomic facts for open-ended generation.

The "non-adversarial + persistent store + cross-session propagation" intersection that looked open in the poisoning survey is precisely what HaluMem + ConsistencyGate occupy.

## 2. Phenomenon weak in our setup (pilot)
`experiments/faithfulness_pilot.py`: LLM-judge support-check of extracted facts vs their source session. **0 / 120 facts unsupported (~0%)** (indicative; job died at 120, single judge qwen3:8b, conservative parse-fallback). Grounded *extractive* memory (facts pulled from a provided transcript) is far more faithful than open-ended generation — so unlike duplication (~10-62%), hallucinated-memory is not even a strong phenomenon here.

## 3. Meta-conclusion (the important one)
Four directions probed, four dead ends:
1. Original hypothesis (canonical units + dedup + multi-view + budget) — not novel (Phase 1-2).
2. Dedup / redundancy — empirically dead (Phase 3).
3. Precision-oriented multi-hop conversational-memory retrieval — pre-empted (EviMem, MRAgent).
4. Memory faithfulness — pre-empted (HaluMem et al.) + weak phenomenon.

**LLM agent memory is a saturated, hyper-active subfield in 2026.** Every first-principles angle is already published within 1-9 months by well-resourced groups. A from-scratch **top-tier method** here is not realistically attainable on this timeline. This is a structural conclusion, not bad luck.

## Realistic outputs from the work done
- **(a) Honest workshop / short empirical-analysis paper** built on the real, reproducible results we have: the bottleneck decomposition (oracle diagnostic — retrieval precision dominates; temporal is a floor even with perfect evidence; full-context distracts) + negative results (blind dedup harmful, redundancy-aware composition neutral) on a small open local model (qwen3:8b). Modest, defensible, workshop/arXiv tier — NOT top-tier method. Must cite HaluMem/EviMem/lost-in-the-middle honestly as related work.
- **(b) Change field entirely** (not just sub-direction within agent memory) if top-tier is non-negotiable — open-ended, no guarantee.
- **(c) Stop** — treat as a completed, rigorous investigation.
