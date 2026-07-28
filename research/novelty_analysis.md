# Novelty Analysis & Go/No-Go Verdict (Phase 2)

Built on full-text reads of the 5 closest works (AdaGReS, MIRIX, Collaborative Memory, Zep, SemDeDup), the [`novelty_matrix.md`](novelty_matrix.md), and a synthetic **coupling pilot** ([`../experiments/pilot_coupling.py`](../experiments/pilot_coupling.py), output in [`../experiments/pilot_coupling_output.txt`](../experiments/pilot_coupling_output.txt)).

---

## TL;DR verdict — **AMBER → GREEN, but the theorem must be reframed**

The pilot did exactly what a gate should: it **falsified the theorem I originally proposed** and surfaced a **different, cleaner, still-novel result**.

- ❌ **The literal "coupling theorem" does NOT hold.** "Deduplication monotonically reduces *both* storage duplication and per-query context tokens" is **trivial for storage and false/composer-dependent for tokens.** Under an optimal composer, dedup leaves per-query tokens unchanged (the composer already avoids redundancy); under a fill-to-budget top-k retriever, tokens also stay ~flat. So a paper built on that monotonicity claim would be **wrong**. Good that we tested before writing it.
- ✅ **A sharper result DID survive, and it is novel and characterizable:** *query-distribution-aware canonicalization Pareto-dominates naive semantic dedup* — naive dedup **loses task utility in proportion to the query-relevant-distinction rate ρ**, while query-aware dedup preserves utility at modest extra storage. And under realistic (sub-optimal) retrievers, dedup improves **task success per token** (efficiency), not raw token count.
- ✅ **The measurement gap is airtight.** None of the 5 closest systems (or the 14 in the matrix) define a duplication-ratio or reuse-rate metric.

**Decision:** proceed toward a top-tier attempt, but pivot the central claim from *"storage↔token coupling"* to *"query-aware canonicalization + the measurement of an axis the field ignores."* See §9.

---

## The pilot, honestly (synthetic — not a benchmark)

Store: 200 facts × 4 near-duplicate units = 800 units, 31,851 stored tokens. ρ = fraction of duplicate clusters that carry a **task-relevant distinction** (merging them loses info some queries need). Numbers are seeded simulation output.

| ρ | dedup | composer | stored tok | ctx tok/q | success |
|--:|---|---|--:|--:|--:|
| 0.00 | none | oracle | 31851 | 138.7 | 1.000 |
| 0.00 | naive | oracle | 5551 | 138.7 | 1.000 |
| 0.00 | none | topk | 31851 | 288.7 | 0.338 |
| 0.00 | naive | topk | 5551 | 288.7 | **0.419** |
| 0.10 | naive | oracle | 5551 | 134.9 | **0.866** |
| 0.10 | query_aware | oracle | 6292 | 141.0 | **1.000** |
| 0.30 | naive | oracle | 5551 | 123.8 | **0.583** |
| 0.30 | query_aware | oracle | 7875 | 145.8 | **1.000** |

**What it shows:**
1. **Storage reduction from dedup is large but trivial** (800→200 units): not a contribution.
2. **Per-query token reduction from dedup is NOT automatic** (138.7 unchanged under oracle; ~288 unchanged under fill-to-budget top-k). The original coupling claim fails here.
3. **Under a realistic top-k retriever, dedup raises success at equal budget** (0.338 → 0.419 at ρ=0): redundant near-dups waste retrieval slots; canonicalization converts wasted tokens into distinct-fact coverage. This is a **token-efficiency** (quality-per-token) gain, not a token-count reduction.
4. **Naive semantic dedup has a real, growing failure mode**: success 1.000 → 0.866 (ρ=0.10) → 0.583 (ρ=0.30). **Query-aware dedup holds success at 1.000** for +13–42% storage over naive — still a 4–5× reduction vs no dedup. This is the defensible, characterizable result.

> Caveats: synthetic; "task-relevant distinction" is idealized; the top-k model fills to budget (a stop-when-covered composer would shift some effect from success into tokens). The pilot informs the *direction*, not the paper's numbers.

---

## The eight questions (Part 5)

**1. What has already been done?**
Canonical units, multi-view typed memory, shared units (via access control), query-aware routing, token-budgeted redundancy-aware selection (with submodular guarantees), context compression, and large token/storage savings. All precedented — see [`novelty_matrix.md`](novelty_matrix.md).

**2. What is the closest prior work?**
- *Optimization core:* **AdaGReS** — `F(q,C)=α·relevance−β·redundancy` s.t. `tokens≤T_max`, greedy, (1−1/e)−kε/e guarantee. But **per-query over a fixed corpus**, homogeneous chunks, **no store co-design, no dedup metric**.
- *Multi-view:* **MIRIX** — six **separate** stores, **no cross-type dedup**, storage claim is SQLite-vs-images.
- *Shared units + reuse:* **Collaborative Memory** — sharing by **access policy, not semantics**; no dedup, no budget.
- *Store-side dedup + tokens:* **Zep** — entity resolution + fact invalidation + token-bounded retrieval, but **incremental (no query-aware store design)** and **no dedup metric**.
- *Semantic dedup:* **SemDeDup** — training data, **destructive, offline, corpus-agnostic**.

**3. What parts of our idea are NOT novel?**
Canonical units; multi-view; token-budgeted redundancy-aware selection (AdaGReS/Lin–Bilmes own this); context compression; reporting token/storage savings; implicit consolidation/merging.

**4. What parts MIGHT be novel? (post-pilot, re-ranked)**
- **(strong) Measuring duplication/reuse** — no prior system does. A metric suite + a redundancy-stress audit stands on its own.
- **(strong) Query-distribution-aware canonicalization** with a characterization of the storage×utility trade-off (utility loss ∝ ρ; query-aware dominates naive). The pilot supports this; it is provable-style and testable.
- **(moderate) Shared-canonical projection vs independent-view storage** (B) — eliminates cross-view duplication MIRIX/Collab incur; measurable.
- **(downgraded) Storage↔token monotonic coupling** — pilot **falsified** it; drop as a headline claim.

**5. What would a skeptical reviewer say?**
- *"AdaGReS already does budgeted redundancy-aware selection."* → We differentiate on **store co-design + typed units + query-aware canonicalization + the utility-loss characterization**, and use AdaGReS as the *inner* composer, not the contribution.
- *"Dedup obviously saves tokens."* → Our own pilot shows it **does not** in general; we reframe to token-efficiency + utility-preservation, which is non-obvious.
- *"MIRIX/Zep already store efficiently."* → Neither measures duplication; we show (with a new metric) how much they silently duplicate, and that query-aware canonicalization preserves utility where naive dedup fails.
- *"Synthetic-only."* → Requires LoCoMo + LongMemEval + a reuse benchmark, ≥2 LLMs.

**6. Smallest defensible novel contribution (revised after pilot):**
> **A query-distribution-aware semantic canonicalization of a live agent-memory store that provably trades storage for task-utility along a characterized frontier (utility loss bounded by the query-relevant-distinction rate), evaluated with the first duplication-ratio / reuse-rate / token-efficiency metric suite for agent memory — showing that naive semantic dedup (à la SemDeDup) silently degrades task utility while query-aware canonicalization does not.**

This is defensible because: (a) it is falsifiable and the pilot already shows the effect; (b) it targets the two empty matrix columns (StoreCoDesign, DupMetric); (c) it turns SemDeDup/AdaGReS into baselines it beats on a *measured* axis; (d) it does **not** rest on the falsified coupling claim.

**7. What additional technical contribution would strengthen it?**
- A **theorem** (now well-posed): bound expected task-utility loss of canonicalization as a function of ρ and the merge threshold τ, and show query-aware canonicalization is Pareto-optimal in {storage, utility} under a stated query model. This is the honest replacement for the coupling theorem.
- A **learned** query-relevance-of-distinction estimator (using FILCO-style CXMI) to set τ per cluster — addresses Cost-Aware Selection's "no static selector dominates."

**8. What experiments establish novelty?**
- **Metric audit (C):** run the new duplication/reuse suite across Mem0/MIRIX/Zep/A-MEM → show unmeasured duplication (novel because nobody has reported it).
- **Utility-frontier (revised A):** naive vs query-aware canonicalization vs no-dedup on LoCoMo/LongMemEval; plot storage × task-utility; show naive loses utility as redundancy/ρ grows, query-aware does not.
- **Efficiency (realistic retriever):** success-per-token under top-k retrieval, dedup vs none.
- **Representation (B):** shared-canonical projection vs MIRIX-style independent views → duplication eliminated at equal accuracy.
- **Generalization:** ≥2 benchmark families, ≥2 LLMs.

---

## 9. Go/No-Go decision

**Verdict: GO — toward a top-tier attempt — with the central claim pivoted.**

| Element | Status after gate |
|---|---|
| Measurement contribution (duplication/reuse metrics + audit) | **Solid.** Empty in all 14 systems. Top-tier viable alone (Datasets & Benchmarks). |
| Query-aware canonicalization + utility-loss characterization | **Solid & novel.** Pilot supports; theorem now well-posed. |
| Shared-canonical projection (vs independent views) | **Solid mechanism.** Differentiates from MIRIX/Collab. |
| Original storage↔token coupling theorem | **Dropped.** Pilot falsified the monotonicity claim. |

> **UPDATE (Phase 3, 2026-07-22) — the method half was empirically tested and did NOT hold.** See [`phase3_pilot_results.md`](phase3_pilot_results.md). On LoCoMo with a proper LLM judge: (i) real semantic duplication exists (~10% at τ=0.85; higher in LLM-built stores); (ii) **naive/blind dedup is harmful** (dominated at every budget); (iii) **query-aware (MMR) composition is statistically indistinguishable from full retrieval** on multi-hop at n=282 (McNemar p=0.90 at k=8, p=0.49 at k=12). The revised-A "query-aware canonicalization beats naive without losing utility" reduces to "query-aware is *safe* but gives *no benefit*." **Revised-A is therefore NOT a viable top-tier method result.** The surviving contribution is **C (measurement) + the negative/analysis finding** — see the addendum at the end of this file.

**Recommended paper (pre-Phase-3, now superseded — kept for the record) = C (measurement) + revised-A (query-aware canonicalization with the ρ-bounded utility theorem) + B (shared-canonical projection as the mechanism).** Headline: *"What agents remember twice: measuring memory redundancy, and query-aware canonicalization that saves storage without the task-utility loss naive dedup incurs."*

**Fallback ladder (unchanged, now evidence-backed):** if the ρ-utility theorem proves hard to state cleanly, **C alone** (metrics + redundancy-stress audit of SOTA) is a credible NeurIPS Datasets & Benchmarks submission, with revised-A as the method section.

**Kill criterion (honesty guard):** if the metric audit shows SOTA systems already carry *negligible* duplication (i.e., their implicit consolidation is near-optimal), the measurement contribution weakens and we must re-scope — this must be reported, not hidden.

---

## 10. Phase 2 summary

- **Completed:** full-text reads of the 5 closest works; novelty matrix (14 systems × 10 dims); coupling pilot built, run, and interpreted; novelty analysis + go/no-go.
- **Learned:** the proposed coupling theorem is false/trivial (pilot-falsified); the durable novelty is **(i)** measuring an axis nobody measures and **(ii)** query-aware canonicalization with a ρ-bounded utility guarantee. Two matrix columns (StoreCoDesign, DupMetric) are empty across all prior work.
- **Uncertain:** whether the ρ-utility theorem states cleanly under a realistic query model; how much duplication real SOTA systems actually carry (the kill-criterion question).
- **Decisions:** drop the coupling-monotonicity headline; adopt the C+revised-A+B framing; keep the fallback to a Datasets & Benchmarks submission.
- **What I need from you:** approve the pivot (measurement + query-aware canonicalization, not storage↔token coupling), and confirm the fallback (benchmark-track) is acceptable if the theorem resists.
- **Recommended next step (Phase 3/4):** write `research_gap.md` → `research_proposal.md` formalizing revised-A's ρ-utility theorem statement and the metric definitions, then a **second, less-synthetic pilot on real LoCoMo data** to measure actual duplication in one SOTA system (turns the kill-criterion into evidence) before committing to full implementation.

---

## Addendum — Phase 3 empirical resolution (2026-07-22)

The method half was tested end-to-end on LoCoMo (local qwen3:8b + nomic-embed, LLM-judged). Results in [`phase3_pilot_results.md`](phase3_pilot_results.md).

**Resolved verdict:**
- ✅ **Measurement contribution stands.** Semantic memory duplication is real (~10% at τ=0.85 in LoCoMo observations; up to ~62% in a naive LLM-built store), pervasive, invisible to lexical methods, and unmeasured by any of the 14 surveyed systems. Metrics MDR/RTF are new to this setting.
- ✅ **Necessity/harm result stands.** Blind semantic dedup is Pareto-dominated at every budget (multi-hop k=8: 0.15 vs full 0.37). "You cannot compact a memory store by blind semantic merging."
- ❌ **Query-aware method does NOT hold.** MMR composition ≈ full retrieval on multi-hop at n=282 (McNemar p=0.90 @k8, 0.49 @k12). Safe, not beneficial. No top-tier method result.

**Realistic paper targets (honest):**
1. **Measurement + negative-result / analysis paper** (recommended). "Duplication in LLM agent memory: pervasive, and neither blind dedup (harmful) nor redundancy-aware composition (no benefit) fixes it." Fits NeurIPS Datasets & Benchmarks or an empirical-analysis / "lessons" track. Real, defensible, reproducible — but not a top-tier *method* main-track paper.
2. **Pivot the research question** if a top-tier *method* is the hard requirement — e.g. away from dedup/redundancy (empirically a dead end here) toward a different memory bottleneck this evidence exposes (retrieval precision and answer reformulation dominate the error budget far more than redundancy does).

**What this evidence does NOT support:** any paper claiming dedup / canonicalization / redundancy-aware composition improves agent-memory task performance. The data says otherwise.
