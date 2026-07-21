# Literature Review — Long-Term Memory for LLM Agents (Phase 1)

**Project:** Canonical reusable memory units + semantic deduplication + cross-context reuse + multi-view memory + token-budgeted context composition.
**Scope:** ~71 papers/systems, emphasis on 2024–2026. Companion machine-readable matrix: [`literature_matrix.csv`](literature_matrix.csv).
**Date compiled:** 2026-07-21.

> **Verification & anti-fabrication note.** Every entry has a real arXiv ID / ACL Anthology ID / DOI and URL. Verification codes in the CSV: `V` = arXiv abstract fetched directly during this review; `A` = fetched by a research sub-agent; `K` = well-established/widely-known work; `P`/preprint flags where peer-review venue is unconfirmed. During the search we encountered **plausibly-fabricated future-dated arXiv IDs** (e.g. `2606.29328`, `2604.26981`, `2603.09222`); these were **excluded**. Any 2025–2026 preprint should still be re-opened before it is cited in the paper. A few reported numeric results (e.g. Reflexion's 91% pass@1, Generative Agents ablations) are widely cited but were **not** re-verified from the PDF and are flagged accordingly — do not quote them as our evidence.

---

## 1. How the field is organized

Two recent surveys give the scaffolding. Zhang et al. 2024 (*A Survey on the Memory Mechanism of LLM-based Agents*, arXiv:2404.13501) organizes agent memory by **sources → forms → operations** (writing, management/reflection, reading) and flags unified evaluation, multimodal/parametric memory, and forgetting/updating as open. Wu et al. 2025 (*From Human Memory to AI Memory*, arXiv:2504.15965) proposes an **object (personal/system) × form (parametric/non-parametric) × time (short/long)** taxonomy yielding eight quadrants. Neither survey centers *deduplication* or *joint storage-and-token optimization* — a first signal about where whitespace may lie.

I group the literature into eight themes below.

---

## 2. Foundational agent-memory architectures

- **MemGPT / Letta** (Packer et al. 2023, arXiv:2310.08560). OS metaphor: a fast in-context "main memory" and slow external "archival" tier, with the LLM paging data in/out via function calls to stay within a fixed window. This is the canonical **token-budgeted composition via paging** reference that nearly every later system benchmarks against. No semantic dedup; no multi-view.
- **Generative Agents** (Park et al. 2023, arXiv:2304.03442, UIST 2023). NL "memory stream" of timestamped observations, with **reflections** synthesized on top; retrieval scores memories by *recency + importance + relevance*. The template for scored, budget-limited memory selection and reflection-based compression.
- **MemoryBank / SiliconFriend** (Zhong et al. 2023, arXiv:2305.10250, AAAI 2024). Dialogue memories + event summaries + a global user-personality profile, with an **Ebbinghaus forgetting-curve** decay for lifecycle management (a distinctive pruning angle, but not semantic dedup).
- **Reflexion** (Shinn et al. 2023, arXiv:2303.11366, NeurIPS 2023). Verbal self-reflections in an episodic buffer, reused across retry episodes — the canonical *memory-of-experience* at the lesson level.

**Takeaway:** the founding systems establish tiered/hierarchical NL memory, scored retrieval, reflection/summarization compression, and paging — but treat each memory as an isolated text blob. None deduplicate semantically or maintain a shared canonical unit reused across views.

---

## 3. Production/extraction-centric memory systems (2024–2026)

- **Mem0 / Mem0^g** (Chhikara et al. 2025, arXiv:2504.19413). Extracts salient NL **facts**; an explicit **ADD/UPDATE/DELETE** consolidation step curbs redundancy; Mem0^g adds a directed labeled graph over entities. Reports LLM-judge +26% over OpenAI's memory, 91% lower p95 latency, >90% token-cost savings on **LoCoMo**. One of the closest systems to "extracted reusable units + consolidation + token savings," with base-vector vs graph as two representations.
- **A-MEM** (Xu et al. 2025, arXiv:2502.12110, NeurIPS 2025). **Zettelkasten notes** — each with contextual description, keywords, tags — dynamically linked; a "memory evolution" step updates linked notes when new ones arrive. The strongest analog to *canonical reusable units with multiple attributes*, though dedup is via linking/evolution rather than an explicit semantic-dedup module.
- **MemoryOS** (Kang et al. 2025, arXiv:2506.06326, EMNLP 2025 Oral). Short/mid/long-term tiers with heat-based promotion + user/agent personas; F1 +49% and BLEU-1 +46% avg over baselines on LoCoMo. A modern OS-style, token-budgeted hierarchical composer.
- **LightMem** (Fang et al. 2025, arXiv:2510.18866, ICLR 2026). Three-stage (sensory→short→long) with **structured memory entries** and an **offline update that detects semantic overlap → conflict detection → LLM knowledge fusion**. Reports token-usage reductions up to 38×/20.9× and API-call reductions up to 30×/55.5× on **LongMemEval**. Among the closest to *canonical structured units + semantic-dedup-via-fusion + token budgeting*.
- **Memory-R1** (Yan et al. 2025, arXiv:2508.19828) and **AgeMem / Agentic Memory** (Yu et al. 2026, arXiv:2601.01885). Both make memory *lifecycle management* a **learned RL policy** (ADD/UPDATE/DELETE/NOOP; store/retrieve/update/summarize/discard). Relevant because dedup/curation could be framed as a learned policy — but neither performs explicit semantic dedup or multi-view composition.
- **RecMem** (Dai et al. 2026, arXiv:2605.16045, Findings of ACL 2026). A lightweight "subconscious" embedding layer defers LLM extraction until **sustained recurrence of semantically similar interactions** is detected — reducing memory-construction token cost by up to 87%. Recurrence ≈ implicit semantic clustering before consolidation.

**Takeaway:** 2024–2026 systems increasingly (a) extract compact units, (b) consolidate/merge or learn lifecycle ops, and (c) report large token savings. Consolidation is common; **explicit, embedding-clustered semantic deduplication with an audited canonical representative is not**.

---

## 4. Graph- and hierarchy-structured memory

- **HippoRAG** (Gutiérrez et al. 2024, arXiv:2405.14831, NeurIPS 2024) and **HippoRAG 2** (2025, arXiv:2502.14802, ICML 2025). Open KG "hippocampal index" + Personalized PageRank retrieval; entity resolution/synonymy provides node-level merging. +up to 20% multi-hop QA; 10–20× cheaper retrieval.
- **Zep / Graphiti** (Rasmussen et al. 2025, arXiv:2501.13956). **Bi-temporal knowledge graph** with episodic nodes, semantic entities/facts (validity intervals), and community summaries. Explicit **entity resolution + fact invalidation** over time and **three tiers of abstraction** (multiple views). DMR 94.8% (vs MemGPT 93.4%); LongMemEval +up to 18.5% with ~90% latency reduction. Among the closest overall to our full framing.
- **GraphRAG** (Edge et al. 2024, arXiv:2404.16130). Entity KG + hierarchical **community summaries** (Leiden), map-reduced for global sensemaking — community summaries are multi-level compressed views.
- **G-Memory** (Zhang et al. 2025, arXiv:2506.07398, NeurIPS 2025 spotlight). **Three-tier graph hierarchy** (insight/query/interaction) for *multi-agent* systems with explicit **cross-trial reuse** and agent-specific customization; +up to 20.89% embodied success.
- **MemTree** (Rezazadeh et al. 2024, arXiv:2410.14052) and **HiGMem** (Cao et al. 2026, arXiv:2604.18349, Findings of ACL 2026). Hierarchical tree / two-level event-turn memory with **multi-granularity views**; HiGMem uses event summaries as semantic anchors to decide which turns to read (anchor-guided, token-reducing retrieval; adversarial F1 0.54→0.78).
- **MemGAS** (Xu et al. 2025, arXiv:2505.19549). **Multi-granularity** memory units associated via Gaussian Mixture Models, with an **entropy-based router** that adaptively selects granularity — directly on the *multi-view + adaptive selection* axis.
- **Cognee** (Markovic et al. 2025, arXiv:2505.24478) / **Memary** (GitHub only). Graph+vector+relational memory frameworks; cite as systems.

**Takeaway:** graph/hierarchical memory delivers *entity-level merging, temporal invalidation, and multi-granularity views* — the closest existing machinery to "shared canonical units + multiple views." Zep, G-Memory, MemGAS, MemTree/HiGMem are the systems to beat/position against.

---

## 5. Multi-view and cross-context / cross-task reuse (the crux)

This is where the proposed idea overlaps most with prior work, so it gets its own section.

- **MIRIX** (Wang & Chen 2025, arXiv:2507.07957). **Six typed memory components** (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault) coordinated by a multi-agent controller; **99.9% storage reduction**; 85.4% on LoCoMo (SOTA claimed). This is an *explicit multi-view typed memory* with composition — much of the "multiple task-specific views" idea already realized.
- **Collaborative Memory** (Rezazadeh et al. 2025, arXiv:2505.18279). Private + shared memory fragments with immutable provenance; **read policies project fragments into filtered *transformed views*** under time-varying access control, enabling **cross-user reuse**. This is the clearest existing instance of *"multiple views over shared memory units + governed cross-context reuse."*
- **Agent Workflow Memory (AWM)** (Wang et al. 2024, arXiv:2409.07429). Induces reusable **workflows** (routines with example-specific context abstracted out) and **selectively injects** them; +24.6% Mind2Web, +51.1% WebArena. Canonical *reusable unit = abstracted workflow*, with budgeted injection and explicit cross-task reuse.
- **ExpeL** (Zhao et al. 2023, arXiv:2308.10144, AAAI 2024). Distills **insights/rules** from experience trajectories and recalls them for unseen tasks (cross-task transfer).
- **Voyager** (Wang et al. 2023, arXiv:2305.16291, TMLR). Ever-growing **skill library** of executable code, retrieved by description embedding and composed for new tasks — *reusable unit = skill*.
- **ALFWorld** (Shridhar et al. 2021, arXiv:2010.03768). Text↔embodied aligned representations of the *same* task — a concrete precedent for "two views of one underlying unit," and a cross-context transfer testbed.
- **PerLTQA** (Du et al. 2024, arXiv:2402.16288). Typed semantic vs episodic personal memory with a classify→retrieve→synthesize pipeline.

**Takeaway (important for novelty):** *multiple views* (MIRIX, Collaborative Memory, MemGAS, PerLTQA), *cross-context reuse* (AWM, ExpeL, Voyager, G-Memory, Collaborative Memory), and *shared memory units* (Collaborative Memory) **already exist, separately and in real 2025 systems.** The proposal's individual pillars are largely precedented. Novelty must come from a specific *combination/mechanism*, not from any single pillar (see [`initial_research_gap.md`](initial_research_gap.md)).

---

## 6. Memory / context compression

- **Prompt/context compression into reusable soft units:** Gist Tokens (Mu et al. 2023, arXiv:2304.08467, up to 26×), AutoCompressor (Chevalier et al. 2023, arXiv:2305.14788), ICAE (Ge et al. 2023, arXiv:2307.06945, 4×), xRAG (Cheng et al. 2024, arXiv:2405.13792, doc→1 token), 500xCompressor (Li et al. 2024, arXiv:2408.03094, 6–500×). These *cache a compressed unit reused across many queries* — the nearest analog to "canonical reusable units," but reuse is by caching, not by merging semantically equivalent memories.
- **Recurrent/compressive long-context memory:** RMT (Bulatov et al. 2022, arXiv:2207.06881), Infini-attention (Munkhdalai et al. 2024, arXiv:2404.07143). Bounded/fixed-parameter memory for long context (paradigm contrast to explicit units).
- **Dialogue consolidation:** Recursively Summarizing (Wang et al. 2023, arXiv:2308.15022) folds history into a running summary.
- **Parametric memory:** MemoryLLM (arXiv:2402.04624, ICML 2024) and M+ (arXiv:2502.00592, ICML 2025) store knowledge in latent pools rather than context — a boundary case for "memory that never enters the token budget."

**Takeaway:** compression is mature and gives strong token savings, but operates on *raw text/activations*, not on *typed, deduplicated, reusable memory units with multiple views*.

---

## 7. Semantic deduplication & conflict resolution

- **SemDeDup** (Abbas et al. 2023, arXiv:2303.09540). The **canonical embedding-based semantic near-duplicate removal** method: embed → k-means cluster → drop within-cluster near-duplicates, keeping one representative. Removes ~50% of LAION with minimal loss. **But it targets training data, not live agent memory.** No agent-memory paper in this survey applies SemDeDup-style dedup with an audited canonical representative to a memory store.
- **Deterministic conflict resolution** (Reddy & Challaram 2026, arXiv:2606.01435). Argues that choosing among contradictory fact versions should be a **deterministic assembly step** (`max(serial)` by recency), not LLM freshness-tracking; +28 points over HippoRAG-v2 on MemoryAgentBench's FactConsolidation. Relevant to *canonicalization by recency*, complementary to semantic dedup.

**Takeaway:** this is the **least-covered dimension**. Consolidation/merging appears implicitly (A-MEM evolution, Mem0 UPDATE, LightMem fusion, Zep invalidation), but *explicit semantic deduplication of the memory store into shared canonical units* is essentially unclaimed.

---

## 8. Token-efficient retrieval & token-budgeted context selection

- **Query-aware retrieved-context compression / pruning:** LongLLMLingua (arXiv:2310.06839), RECOMP (arXiv:2310.04408, ICLR 2024), FILCO (arXiv:2311.08377 — its **CXMI** per-sentence utility is directly reusable as a memory-unit utility signal), Provence (arXiv:2501.16214, ICLR 2025 — unifies reranking + adaptive-amount pruning), EXIT (arXiv:2412.12559, Findings ACL 2025). Task-agnostic token classifiers: LLMLingua / LLMLingua-2 (arXiv:2310.05736 / 2403.12968), Selective Context (arXiv:2310.06201).
- **Adaptive cardinality / effort:** Adaptive-k (arXiv:2506.08479, EMNLP 2025 — "how many to include" via similarity-gap), Adaptive-RAG (arXiv:2403.14403, NAACL 2024 — route no/single/multi retrieval), Self-RAG (arXiv:2310.11511, ICLR 2024), FLARE (arXiv:2305.06983, EMNLP 2023).
- **Explicit budget-constrained set selection (most relevant):**
  - **AdaGReS** (Peng et al. 2025, arXiv:2512.25052). **The closest existing formalization of the proposal's optimization core:** a set-level objective = query relevance − intra-set redundancy, **greedy selection under a token-budget constraint**, with an instance-adaptive trade-off calibration and **ε-approximate submodularity / near-optimality guarantees**. Operates on RAG chunks with embedding relevance (not typed memory units, not measured task utility).
  - **Cost-Aware Evidence Selection** (Wu et al. 2026, arXiv:2606.02245). Benchmarks budget-constrained selectors (top-k / relevance-only / cost-aware greedy / knapsack / redundancy-aware) under an access budget; key **negative result: no static selector dominates and larger budgets don't reliably help** — strong motivation for a *learned/adaptive* budgeted policy.
  - **Lin & Bilmes 2011** (ACL P11-1052; + NAACL 2010 budgeted variant). The classical **submodular coverage-minus-redundancy maximization under a budget**, solved greedily with (1−1/e)/cost-scaled guarantees — the theory a token-budgeted memory composer would instantiate.

**Takeaway:** the "token-budgeted composition of the smallest sufficient set" idea is **substantially precedented** — AdaGReS already does budget-constrained relevance−redundancy greedy selection with submodular guarantees; Lin–Bilmes is the 2011 theoretical template. What is *not* done: applying this to **heterogeneous, deduplicated, typed memory units** with **measured task utility** rather than embedding relevance over RAG chunks.

---

## 9. Evaluation landscape

- **Long-term conversational memory:** LoCoMo (arXiv:2402.17753, ACL 2024; the de-facto standard), LongMemEval (arXiv:2410.10813, ICLR 2025; its index→retrieve→read decomposition + token analysis map directly onto our composition), MSC (arXiv:2107.07567, ACL 2022), Conversation Chronicles (arXiv:2310.13420), DialSim (arXiv:2406.13144).
- **Memory-agent competency:** MemoryAgentBench (Hu et al. 2025, arXiv:2507.05257) — four competencies: accurate retrieval, **test-time learning**, long-range understanding, **selective forgetting**. Its FactConsolidation task is the conflict-resolution testbed used by Reddy & Challaram 2026.
- **Personalization:** LaMP (arXiv:2304.11406, ACL 2024), LongLaMP (arXiv:2407.11016).
- **Long-horizon agents (reuse substrate):** WebArena (arXiv:2307.13854), Mind2Web, ALFWorld, τ-bench (arXiv:2406.12045), AgentBench (arXiv:2308.03688), GAIA (arXiv:2311.12983).

**Takeaway:** LoCoMo + LongMemEval + MemoryAgentBench are the natural primary benchmarks; WebArena/Mind2Web (via AWM) and ALFWorld are the natural *cross-task-reuse* testbeds; LaMP/PerLTQA cover personalization. Crucially, standard benchmarks measure **answer quality** but under-measure **storage duplication, memory reuse rate, and token efficiency jointly** — an evaluation gap we can also contribute to.

---

## 10. Consolidated relevance map

| Proposed pillar | Strongest existing precedents | Status |
|---|---|---|
| Canonical reusable memory units | A-MEM (notes), AWM (workflows), Voyager (skills), Gist/xRAG (cached compressed units), Mem0 (facts) | **Precedented** |
| Semantic deduplication of memory | SemDeDup (training data, not memory); implicit merging in A-MEM/Mem0/LightMem/Zep | **Largely open for live agent memory** |
| Shared memory representation | Collaborative Memory (shared fragments), Zep (shared graph) | **Precedented** |
| Multiple memory views | MIRIX (6 types), Collaborative Memory (transformed views), MemGAS/MemTree (granularity), PerLTQA (typed) | **Precedented** |
| Cross-context / cross-task reuse | AWM, ExpeL, Voyager, G-Memory, Collaborative Memory | **Precedented** |
| Query-aware routing | MemGAS (entropy router), Adaptive-RAG (complexity router), Self-RAG | **Precedented** |
| Token-budgeted selection | **AdaGReS** (budget + redundancy + submodular), Lin–Bilmes, Cost-Aware Selection, Adaptive-k | **Strongly precedented** |
| Context composition/compression | LLMLingua family, RECOMP, Provence, MemGPT paging | **Precedented** |
| **Joint optimization of storage duplication AND context tokens** | *No single paper optimizes both jointly with a shared canonical-unit store* | **Apparent whitespace** |

---

## 11. Headline findings

1. **Every individual pillar of the original hypothesis is already precedented** — often in real 2025 systems (MIRIX, Collaborative Memory, MemGAS, AWM, Zep, LightMem, AdaGReS). The idea is **not novel as a bundle of features**.
2. **The token-budgeted-selection core is the most contested:** AdaGReS (2025) already formalizes budget-constrained relevance−redundancy greedy selection with submodular guarantees; Lin–Bilmes provided the template in 2011. A generic "select the smallest sufficient set under a budget" contribution would likely be rejected as incremental.
3. **The least-covered dimension is explicit semantic deduplication of a *live agent memory store* into shared canonical units** — SemDeDup exists but for training data, and agent-memory systems only merge implicitly.
4. **No system jointly optimizes storage duplication and context-token cost over a shared canonical-unit store, nor measures both** — the evaluation literature under-reports duplication/reuse metrics.
5. **The most defensible novel angle is therefore an *integration + mechanism + measurement* contribution**, not a feature-existence claim. Candidate framings and the precise gap are developed in [`initial_research_gap.md`](initial_research_gap.md).

*Full per-paper fields (extraction/storage/retrieval/limitations/etc.) are in [`literature_matrix.csv`](literature_matrix.csv). Novelty scoring per dimension is deferred to Phase 2 (`novelty_matrix.md`, `novelty_analysis.md`).*
