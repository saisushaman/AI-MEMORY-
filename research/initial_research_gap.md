# Initial Research Gap Analysis (Phase 1 output)

**Purpose:** an honest, evidence-grounded assessment of whether the original hypothesis is novel, where the real gap is, and what a defensible contribution could look like. This is *not* the final research proposal (that is Phase 4, after the formal novelty matrix in Phase 2).

**Bottom line up front:** the original hypothesis, *as stated*, is **largely not novel**. Each of its pillars is precedented — several in real 2025 systems — and its token-budgeted-selection core is already formalized (AdaGReS 2025; Lin–Bilmes 2011). A publishable contribution must move from "we combine these features" to a **specific mechanism + measurement** that no prior work provides. The most defensible whitespace is the **joint optimization of storage-side semantic duplication and context-side token cost over one shared canonical-unit store, with a metric suite that actually measures duplication and reuse.**

---

## 1. What has already been done? (mapped to the original claims)

The original hypothesis has five implicit claims. Verdict on each (evidence in [`literature_review.md`](literature_review.md) / [`literature_matrix.csv`](literature_matrix.csv)):

| Claim | Verdict | Key precedents |
|---|---|---|
| "Information can be represented as canonical, reusable memory units" | **Done** | A-MEM (linked notes), AWM (workflows), Voyager (skills), Mem0 (facts), Gist/xRAG (cached compressed units) |
| "…shared across multiple task-specific memory views" | **Done** | MIRIX (6 typed views), Collaborative Memory (per-user *transformed views* over shared fragments), MemGAS (granularity views), PerLTQA (typed) |
| "A query-aware memory controller dynamically selects…" | **Done** | MemGAS (entropy router), Adaptive-RAG (complexity router), Self-RAG, Provence |
| "…the smallest sufficient set of memory units under a token budget" | **Done / formalized** | **AdaGReS** (budget-constrained relevance−redundancy greedy + submodular guarantee), **Lin–Bilmes 2011** (budgeted submodular coverage−redundancy), Cost-Aware Evidence Selection, Adaptive-k |
| "…reducing context token usage and memory duplication while maintaining task performance" | **Partially done** | Token reduction is heavily demonstrated (Mem0 >90%, LightMem up to 38×, MIRIX 99.9% storage, RecMem 87%). *Duplication reduction is asserted implicitly via consolidation; almost never measured as a first-class metric.* |

**Consequence:** a reviewer can point to a single prior paper for every pillar. The bundle is not itself a contribution.

---

## 2. What is the closest prior work?

Four systems/papers are close enough that the paper must explicitly differentiate from them:

1. **AdaGReS** (arXiv:2512.25052, 2025) — *closest on the optimization core.* Already does token-budgeted greedy selection maximizing relevance minus intra-set redundancy, with adaptive trade-off calibration and ε-submodular near-optimality guarantees. **This nearly pre-empts "token-budgeted selection of a non-redundant sufficient set" as a standalone contribution.** Differences we can exploit: it operates on *RAG chunks*, uses *embedding relevance* (not measured task utility), has *no persistent memory store*, *no deduplication of the store*, and *no multi-view typing*.
2. **MIRIX** (arXiv:2507.07957, 2025) — *closest on multi-view.* Six typed memory components with a coordinating controller and 99.9% storage reduction. Differences: typing is fixed and hand-designed; no explicit semantic dedup into canonical units; no budget-constrained composition objective; storage reduction is reported but duplication is not measured.
3. **Collaborative Memory** (arXiv:2505.18279, 2025) — *closest on shared units + views + cross-context reuse.* Shared fragments projected into per-user *transformed views* under access policies, enabling cross-user reuse. Differences: views come from *access control*, not task/query semantics; dedup and token-budget composition are not the focus.
4. **Zep / Graphiti** (arXiv:2501.13956, 2025) — *closest on a shared canonical store with dedup + tiers.* Bi-temporal graph with entity resolution, fact invalidation, and three abstraction tiers. Differences: dedup = entity resolution + temporal supersession (not embedding-cluster canonicalization); no budget-constrained composition objective; duplication/reuse not measured.

Runner-ups to differentiate from: **A-MEM** (evolving linked notes), **LightMem** (dedup-via-fusion + token savings), **MemGAS** (multi-granularity + adaptive selection), **AWM** (reusable workflows), **SemDeDup** (semantic dedup, but for training data).

---

## 3. What parts of our idea are NOT novel?

- Extracting compact/canonical memory units. (A-MEM, Mem0, AWM, Voyager)
- Maintaining multiple typed/granularity views. (MIRIX, MemGAS, PerLTQA, Collaborative Memory)
- A shared memory store reused across sessions/users/agents. (Zep, Collaborative Memory, G-Memory)
- Query-aware routing/selection. (MemGAS, Adaptive-RAG, Self-RAG)
- Token-budgeted, redundancy-aware set selection with optimality guarantees. (**AdaGReS, Lin–Bilmes** — this is the big one)
- Context compression to save tokens. (LLMLingua family, RECOMP, Provence, Gist/xRAG)
- Reporting large token/storage savings. (Mem0, LightMem, MIRIX, RecMem)

**If the paper's contribution is any one of these, it is not publishable at a top venue.**

---

## 4. What parts MIGHT be novel? (candidate whitespace, to be pressure-tested in Phase 2)

Ranked by how defensible they currently look:

1. **Joint storage-and-context objective over one shared canonical-unit store (strongest).** Prior work optimizes *either* storage (dedup/consolidation: SemDeDup, Zep, LightMem) *or* context tokens (AdaGReS, LLMLingua, RECOMP) — **not both jointly, and not over the same canonical units.** A formulation where deduplicating the store into canonical units is the *same* representation that a budgeted composer draws views from — so that dedup provably shrinks both storage duplication *and* per-query token cost — appears unclaimed. This reframes the contribution from "another selector" to "a representation that makes storage and context efficiency the same optimization."
2. **Explicit semantic deduplication of a *live agent memory store* into audited canonical units (moderate–strong).** SemDeDup does embedding-cluster canonicalization for training data; agent-memory systems only merge implicitly (A-MEM evolution, Mem0 UPDATE, Zep invalidation). Bringing SemDeDup-style canonicalization *into* agent memory, with lossless-view reconstruction and conflict handling (à la Reddy & Challaram 2026), is a concrete, testable mechanism — but it risks being seen as "SemDeDup applied to memory," so it needs the joint-objective framing above to be more than an application.
3. **Task-utility (not embedding-relevance) as the composition objective, with a measurement suite (moderate).** AdaGReS/Cost-Aware Selection use relevance/coverage proxies; Cost-Aware's negative result ("no static selector dominates") invites a *learned, utility-grounded* budgeted composer over typed units. Pairs naturally with new **duplication-ratio / reuse-rate / token-efficiency** metrics that the field currently under-reports.
4. **Cross-view canonical sharing that provably reduces duplication (moderate, depends on #1).** One canonical unit instantiated into many task-specific views (rather than duplicated per view) — the "shared" in the hypothesis — is only novel if we *measure* the duplication avoided and show it is not achievable by existing multi-view systems (which store views largely independently).

---

## 5. What would a skeptical reviewer say?

Anticipated reviews (full simulation is Phase 15; these guide Phases 2–6):

- **"This is MIRIX + AdaGReS + SemDeDup stapled together."** → Must show a *mechanism* where the pieces are inseparable (dedup and budgeted composition sharing one canonical representation), not a pipeline of known parts.
- **"AdaGReS already does token-budgeted redundancy-aware selection with guarantees; what's new in your selector?"** → Differentiate on *heterogeneous typed memory units + measured task utility + a persistent deduplicated store*, and ideally a *joint* objective AdaGReS does not have.
- **"Your token savings just replicate Mem0/LightMem/MIRIX."** → Token savings alone are table stakes. The novelty must be the **duplication↔token coupling** and its measurement, shown against these exact baselines.
- **"You never measure duplication or reuse — the central claims."** → We must define and report **Memory Duplication Ratio, Memory Reuse Rate, Token Efficiency, Context Compression Ratio** (Phase 9), or the paper's own thesis is unfalsified.
- **"Gains are LoCoMo-only / one LLM."** → Need ≥2 benchmark families (e.g. LoCoMo + LongMemEval + a reuse benchmark like WebArena/Mind2Web via AWM) and ≥2 LLMs (RQ6).

---

## 6. Smallest defensible novel contribution (working hypothesis for Phase 4)

> **A shared canonical-memory-unit store in which semantic deduplication and token-budgeted, task-utility-driven view composition are two projections of a single objective — so that consolidating the store provably reduces *both* storage duplication and per-query context tokens — evaluated with a duplication/reuse/token-efficiency metric suite that prior memory systems do not report.**

This is defensible because: (a) it is a *mechanism/representation* claim, not a feature list; (b) it targets the one dimension (explicit dedup of live memory) that is genuinely thin; (c) it turns AdaGReS/Lin–Bilmes from competitors into the *composition sub-routine* inside a larger, novel storage-context-coupled objective; (d) it is directly falsifiable via new metrics.

**What would strengthen it (additional technical contribution):**
- A formal result that, under stated assumptions, deduplication of the canonical store is *monotone* in both storage-duplication reduction and expected per-query token reduction (i.e. the two objectives are aligned, not traded off) — this would be a genuine theoretical hook AdaGReS/MIRIX lack.
- A *learned* utility/sufficiency estimator (leveraging FILCO's CXMI-style signal) so composition uses measured task utility, addressing Cost-Aware Selection's "no static selector dominates."

---

## 7. Experiments needed to establish novelty (preview of Phases 6/9/10)

1. **Baselines that pin each pillar:** full-context, naive vector RAG, summarization memory (Recursive Summarization), Mem0, MIRIX (multi-view), Zep (graph+dedup), AdaGReS (budgeted selection) — showing none jointly optimize duplication + tokens.
2. **The coupling result, measured:** report Memory Duplication Ratio and per-query token cost *as the store grows*, demonstrating our canonical-unit store reduces both where baselines reduce at most one.
3. **Ablations isolating the novel bit:** remove dedup → duplication + tokens rise; remove the joint objective (dedup then independent selection) → show the coupling matters; vs. store views independently (MIRIX-style) → show shared canonical units avoid duplication.
4. **Generalization:** ≥2 benchmark families, ≥2 LLMs (RQ6).
5. **Falsification guard:** if dedup does *not* reduce token cost, or joint ≈ decoupled, the central claim fails honestly — this must be reportable.

---

## 8. Open uncertainties to resolve before Phase 4

- **Is the storage↔token coupling actually non-trivial,** or does any dedup automatically reduce tokens? (Needs a small pilot / formal argument — this determines whether the strongest angle survives.)
- **Does AdaGReS's PDF contain anything closer** (e.g. typed units, task utility) than its abstract suggests? → read the full PDF in Phase 2.
- **Which 2026 preprints are peer-reviewed** vs. arXiv-only, and are any *even closer* (re-verify the fabrication-flagged space carefully).
- **Metric standardization:** are Duplication Ratio / Reuse Rate already defined anywhere? (Not found so far — likely a contribution, but must confirm.)

---

## Phase 1 summary

- **Completed:** ~71 verified papers reviewed; literature matrix + narrative review built; per-pillar novelty pre-assessment done.
- **Learned:** the original hypothesis is a bundle of already-precedented pillars; the token-budget core is pre-empted by AdaGReS/Lin–Bilmes; the genuine thin spot is *explicit semantic dedup of live memory* and, above it, *joint storage-context optimization over shared canonical units* — plus an under-served *measurement* gap (duplication/reuse metrics).
- **Uncertain:** whether the storage↔token coupling is non-trivial enough to anchor a theorem; whether any un-read PDF (esp. AdaGReS) is closer than believed.
- **Decisions made:** do **not** frame novelty as "canonical units + multi-view + budgeted selection" (all precedented); pivot toward the coupling + measurement framing.
- **What I need you to do:** confirm the direction (see recommendation) and whether to prioritize the *theoretical coupling* angle vs. the *empirical dedup-for-memory* angle.
- **Recommended next step:** proceed to **Phase 2 (novelty matrix + analysis)**, including reading the AdaGReS, MIRIX, and Collaborative Memory PDFs in full to lock the differentiation, before committing to a direction in Phase 4.
