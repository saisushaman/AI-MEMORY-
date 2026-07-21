# Research Gap → Opportunity Map (Phase 1 output)

**Purpose:** turn the literature findings into *useful, buildable, top-tier-viable* research directions. For each: what's genuinely new, the exact prior work it must beat and *how*, the claim to prove, the decisive experiment, the risk, and venue fit. Evidence lives in [`literature_review.md`](literature_review.md) / [`literature_matrix.csv`](literature_matrix.csv).

**Frame:** the original hypothesis's *pillars* are precedented — so we stop treating "canonical units + multi-view + budgeted selection" as the contribution and instead compete on **mechanism, guarantee, and measurement**. Below, the same raw ingredients are re-pointed at four angles where the field is actually thin. Each is written so you can decide *fast* whether it's worth building.

---

## The one-paragraph reframe

Every existing memory system treats the **store** and the **per-query context** as two separate problems: dedup/consolidation happens *in* the store (SemDeDup, Zep, LightMem, Mem0), and token-budgeted selection happens *at query time over a fixed store* (AdaGReS, RECOMP, Provence). **No one makes the canonical store itself the decision variable that is optimized against the distribution of future queries and views.** That single shift — *co-designing what you store with how you'll compose it* — is the thread that runs through all four directions below and is where the defensible novelty lives.

---

## Precedent map (compressed — so we differentiate precisely, not vaguely)

| Ingredient | Best prior art | The precise thing it does **not** do |
|---|---|---|
| Budgeted redundancy-aware selection | **AdaGReS** (2512.25052) | Optimizes *per query over a fixed corpus of RAG chunks*; never redesigns the store; no typed units; embedding relevance, not task utility |
| Multi-view typed memory | **MIRIX** (2507.07957) | Stores views as *independent* typed memories → the same fact is duplicated across types; no shared canonical unit; no budget objective |
| Shared units + views + reuse | **Collaborative Memory** (2505.18279) | Views come from *access policy*, not query/task semantics; not about dedup or token budget |
| Semantic dedup | **SemDeDup** (2303.09540) | Training-data only; destructive; no views, no query-time composition, no provenance |
| Store-side consolidation | Zep / LightMem / Mem0 | Entity/fact merging or LLM fusion, but **duplication is never measured** and store isn't optimized w.r.t. query distribution |

**Reusable takeaway:** we don't need a new selector or a new "memory type." We need to make **canonicalization a query-aware, measured, optimized object.**

---

## Direction A — *Canonical memory as a shared decision variable* (co-design of dedup + composition) — **RECOMMENDED**

**Perspective:** optimization / theory.

**Title (working):** "One Store, Many Views: Co-Designing Semantic Deduplication and Token-Budgeted Composition for LLM Agent Memory."

**Core hypothesis.** If the memory store is a set of **canonical units** plus cheap **view-projection functions** (not independently stored views), then choosing *which units to canonicalize* can be optimized against the *distribution of future queries/views* so that a single act of deduplication reduces **both** storage duplication **and** expected per-query context tokens — provably, not incidentally.

**Why it's new (vs the closest work):**
- vs **AdaGReS**: AdaGReS solves `select subset | fixed corpus, this query`. We solve `design canonical store | query distribution` and *then* let an AdaGReS-style composer run on top. The store is the variable; AdaGReS becomes our inner loop, not our competitor.
- vs **MIRIX**: MIRIX duplicates a fact across typed views; we store it once and project. This is *measurably* less duplication — a head-to-head we can win on a metric MIRIX never reports.
- vs **SemDeDup**: SemDeDup dedups blindly (offline, corpus-agnostic, destructive). Ours is *query-distribution-aware and non-destructive* (views reconstruct distinctions on demand).

**Claim to prove (the moat).** *Coupling theorem (to be proven, currently a conjecture):* under stated conditions on the view-projection functions and query distribution, merging semantically equivalent units is **monotone** in both (i) storage-duplication reduction and (ii) expected per-query token cost — i.e. the two efficiencies are aligned, and there exists a characterizable regime where they are **not** (where naive dedup hurts a task by collapsing a needed distinction). Characterizing that boundary is the paper.
> Honesty flag: this theorem is *hypothesized*, not established. Phase 2 must sketch and stress-test it. If it's trivially always-true or false, Direction A downgrades to Findings-tier and we prefer C.

**Decisive experiment.** As the store grows over multi-session data, plot **duplication ratio** and **per-query tokens** for: full-context, Mem0, MIRIX, Zep, AdaGReS-over-fixed-store, and ours. Win = ours reduces *both* where each baseline reduces at most one, at equal or better task accuracy.

**Risk:** the coupling may be trivial (any dedup cuts tokens). Mitigation: the *task-utility* side — show naive dedup *loses accuracy* by merging task-relevant distinctions, and that query-aware canonicalization avoids it. That asymmetry is itself the result.

**Venue fit:** NeurIPS / ICLR (theory + systems). **Top-tier if the theorem has teeth; Findings if not.**

---

## Direction B — *Content-addressed canonical store with on-demand view projection* (representation)

**Perspective:** representation / systems.

**Title (working):** "Project, Don't Duplicate: Content-Addressed Canonical Memory with Query-Conditioned Views."

**Core hypothesis.** Factor every memory into an **invariant canonical content unit** + a set of **view functions** (episodic / semantic / procedural / temporal renderings) computed on demand. One physical unit, many logical views. This is the representational payoff Direction A assumes.

**Why it's new:** MIRIX/PerLTQA/Collaborative Memory all realize "multiple views," but by *storing* them separately. The novelty is the **factorization** — content-addressing (a unit is keyed by its semantic content, so equivalent memories collide and dedup for free) + view-functions that regenerate type-specific presentations. Nobody stores once and projects N ways with measured duplication elimination.

**Claim to prove:** view-projection preserves task accuracy vs. independently-stored views (no information loss) while eliminating cross-view duplication — i.e. "shared canonical + projection" is a *lossless* compression of "N independent typed stores."

**Decisive experiment:** reproduce MIRIX's six-type setup, then swap independent typed stores for one canonical store + projections; measure duplication eliminated and accuracy retained on LoCoMo/LongMemEval.

**Risk:** projection functions may be expensive or lossy. Mitigation: cache hot views; report the compute/latency trade-off honestly (Part 9 metrics).

**Venue fit:** strong systems/Findings; becomes top-tier *only bundled with A's theorem*. **Best used as the mechanism inside Direction A, not alone.**

---

## Direction C — *A duplication-aware benchmark + metric suite that exposes hidden redundancy in SOTA memory* (measurement)

**Perspective:** evaluation. (Benchmark/analysis papers get into top venues — LoCoMo, LongMemEval, MemoryAgentBench all did.)

**Title (working):** "How Much Do Agents Remember Twice? Measuring and Stressing Redundancy in LLM Memory Systems."

**Core hypothesis.** Current SOTA memory systems silently store and re-inject large amounts of semantically duplicate content, and *no benchmark measures it.* A redundancy-stress benchmark (near-duplicate facts recurring across sessions, paraphrased restatements, contradictory updates) plus formal metrics will reveal this and reorder the leaderboard.

**Why it's new:** every benchmark (LoCoMo, LongMemEval, MemoryAgentBench) measures answer quality; **none** measure duplication ratio, cross-view reuse rate, or token efficiency *as a function of redundancy in the input stream*. This is an unclaimed, high-leverage gap.

**Contribution:** (1) mathematically-defined **Memory Duplication Ratio, Memory Reuse Rate, Token Efficiency, Context Compression Ratio**; (2) a redundancy-stress dataset (built by injecting controlled near-duplicates into existing multi-session data — no fabrication of results, just controlled data construction); (3) an audit of Mem0/MIRIX/Zep/A-MEM showing where they duplicate.

**Decisive experiment:** the audit *is* the result — run the suite across SOTA systems, publish the duplication leaderboard.

**Risk:** lower "wow" than a new method. Mitigation: pair with Direction A/B as the evaluation half of a method paper (strongest combo), or stand alone as a Datasets & Benchmarks track submission (a real top-tier track).

**Venue fit:** NeurIPS Datasets & Benchmarks, or the eval backbone of A. **Lowest-risk route to a real top-tier acceptance.**

---

## Direction D — *Non-destructive semantic canonicalization with provenance and reversible merges* (lifecycle)

**Perspective:** dynamics / safety.

**Title (working):** "Reversible Memory: Provenance-Preserving Semantic Deduplication for Agents."

**Core hypothesis.** Dedup is usually destructive (merge → lose the originals → can't recover a distinction a later task needs). A canonical store that keeps provenance and supports **un-merge on demand** (triggered when a query needs a collapsed distinction) beats both no-dedup (bloated) and destructive dedup (lossy), and integrates deterministic conflict resolution (Reddy & Challaram 2026) for contradictory updates.

**Why it's new:** SemDeDup and store-consolidation are destructive; A-MEM "evolution" and Zep "invalidation" transform without a reversible, query-triggered un-merge. Reversibility as a first-class memory operation is unclaimed.

**Claim to prove:** reversible canonicalization dominates the Pareto frontier of {storage, tokens, task accuracy} vs. no-dedup and destructive-dedup.

**Risk:** engineering-heavy; novelty is narrower. **Best as a component/ablation of A, or a follow-up paper.**

**Venue fit:** Findings/workshop alone; a strong *section* of A.

---

## Recommended bet

**Build Direction A, with B as its mechanism and C as its evaluation.** That single paper is:
- **Theoretically novel** (the coupling result — the moat AdaGReS/MIRIX lack),
- **Mechanistically novel** (project-don't-duplicate canonical store — beats MIRIX on a metric it never reports),
- **Empirically novel** (duplication/reuse metrics + redundancy-stress audit the field doesn't have).

Any one of the three alone is Findings-tier. **Together they're a coherent top-tier story: "we reframe memory efficiency as one optimization over a shared canonical store, prove when storage and context efficiency align, and measure a redundancy axis prior work ignored."**

Directions C and D are also your **fallback ladder**: if A's theorem collapses in Phase 2, C alone is still a credible top-tier Datasets & Benchmarks submission, and D backfills a strong methods section.

---

## The go/no-go gate (Phase 2) — what makes this decision, not a guess

Phase 2 answers exactly one question and picks the path:

1. **Read in full** (not abstracts): AdaGReS, MIRIX, Collaborative Memory, SemDeDup, Zep — confirm none already own the coupling result or the metrics.
2. **Sketch the coupling theorem** and stress-test with a toy pilot: does query-aware dedup reduce tokens *without* losing task-relevant distinctions, and is there a characterizable failure regime?
3. **Decide:**
   - Theorem has teeth → **A+B+C, aim top-tier.**
   - Theorem trivial/false → **C standalone (Datasets & Benchmarks) + D**, aim top-tier via the eval route instead.

Either branch lands somewhere real. That's why the gate is worth running before committing weeks.

---

## Mapping back to your original RQs (nothing wasted)

- RQ1 (reuse one unit across views) → **Direction B** (project-don't-duplicate) makes it *measurable*.
- RQ2 (semantic dedup) → **Directions A/D** (query-aware + reversible), not blind SemDeDup.
- RQ3 (views vs flat retrieval) → **Direction C** metrics quantify it.
- RQ4 (token-budget without quality loss) → **Direction A**'s composer (AdaGReS as inner loop) + task-utility objective.
- RQ5 (tradeoffs) → **Direction C**'s Pareto/metric suite is exactly this.
- RQ6 (generalization) → experimental requirement across ≥2 benchmark families, ≥2 LLMs, in every direction.

**Your instinct was right about the ingredients; the novelty was in the wrong place. Move it from "we have these features" to "we prove storage and context efficiency are one optimization, and we measure the axis everyone skipped."**
