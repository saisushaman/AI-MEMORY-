# Novelty Matrix (Phase 2)

Ten proposed dimensions × the systems closest to our idea. A cell is marked **only** if the paper actually supports it (verified from full text where noted in Phase 2; abstract/matrix otherwise). Legend: **●** full support · **◐** partial/implicit · **○** not supported · **—** N/A.

Dimensions:
1. **Canon** — canonical, atomic reusable memory units
2. **SemDedup** — *explicit* embedding-based semantic deduplication of the store
3. **Shared** — one shared unit reused across views (not duplicated per view)
4. **Multi-view** — multiple task/type-specific views
5. **DynView** — views built dynamically from query/task semantics (not fixed types or access rules)
6. **Routing** — query-aware routing/selection
7. **Budget** — hard token-budget-constrained composition objective
8. **Compose** — context composition/compression
9. **StoreCoDesign** — the *store itself* is optimized against the query distribution (not just per-query selection over a fixed store)
10. **DupMetric** — defines/measures a duplication-ratio or reuse-rate metric

| System (year) | Canon | SemDedup | Shared | Multi-view | DynView | Routing | Budget | Compose | StoreCoDesign | DupMetric |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MemGPT (2023) | ◐ | ○ | ○ | ◐ | ○ | ◐ | ● | ● | ○ | ○ |
| Generative Agents (2023) | ◐ | ○ | ○ | ◐ | ○ | ● | ◐ | ● | ○ | ○ |
| Mem0 / Mem0^g (2025) | ● | ◐ | ○ | ◐ | ○ | ● | ◐ | ● | ○ | ○ |
| A-MEM (2025) | ● | ◐ | ◐ | ◐ | ○ | ● | ○ | ◐ | ○ | ○ |
| MemoryOS (2025) | ● | ◐ | ○ | ◐ | ○ | ● | ● | ● | ○ | ○ |
| **Zep / Graphiti (2025)** | ● | ◐ | ◐ | ● | ○ | ● | ● | ● | ○ | ○ |
| LightMem (2025) | ● | ◐ | ○ | ○ | ○ | ● | ● | ● | ○ | ○ |
| MemGAS (2025) | ● | ◐ | ○ | ● | ◐ | ● | ◐ | ◐ | ○ | ○ |
| **MIRIX (2025)** | ● | ○ | ○ | ● | ○ | ● | ○ | ● | ○ | ○ |
| **Collaborative Memory (2025)** | ● | ○ | ● | ● | ○ | ◐ | ○ | ◐ | ○ | ○ |
| AWM (2024) | ● | ◐ | ◐ | ○ | ○ | ● | ◐ | ● | ○ | ○ |
| **AdaGReS (2025)** | ○ | ● | ○ | ○ | ○ | ● | ● | ● | ○ | ○ |
| SemDeDup (2023) | ● | ● | — | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Lin–Bilmes (2011) | ◐ | ◐ | ○ | ○ | ◐ | ◐ | ● | ● | ○ | ○ |
| **Proposed (target)** | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |

## What the columns reveal (verified in Phase 2 full-text reads)

- **DupMetric — nobody has it.** Every one of the 14 systems, including all 5 read in full, reports accuracy/latency/tokens/storage but **none define a duplication-ratio or cross-view reuse-rate metric.** MIRIX's "99.9% storage reduction" is SQLite-vs-raw-images, not a dedup metric. This column is **empty for all prior work** → the cleanest, most defensible contribution.
- **StoreCoDesign — nobody has it.** AdaGReS is explicit: it selects a subset *per query over a fixed corpus* and "does not optimize/redesign the store." Zep builds "incrementally… regardless of queries." No system optimizes the canonical store against the query distribution.
- **SemDedup — only AdaGReS (per-query redundancy) and SemDeDup (training data) truly have it.** Agent-memory systems (Mem0/A-MEM/Zep/LightMem) do *implicit* merging/consolidation (◐), not embedding-cluster canonicalization of the live store. SemDeDup has it but for static training data, destructive, corpus-agnostic.
- **Shared — only Collaborative Memory (●).** But its sharing is governed by *access control*, not semantics, and it has no dedup or budget. MIRIX/MemGAS/Zep keep views as separate structures (○/◐).
- **DynView — essentially nobody (◐ at most).** MIRIX's types are fixed and hand-designed; Collaborative Memory's views are permission filters; MemGAS routes over pre-built granularities. No system *constructs* a view from query/task semantics over shared canonical units.
- **Budget + Compose — well covered** (AdaGReS, Zep, MemoryOS, LightMem, MemGPT). This is *not* where novelty lives.

## Reading

The proposed row differs from every prior row in the **last two columns (StoreCoDesign, DupMetric)** and in the **combination** of Shared + SemDedup + DynView that no single system holds. Novelty must be argued on **those** columns — not on Canon/Multi-view/Budget/Compose, which are saturated.

> Phase-2 caveat: cells for AdaGReS, MIRIX, Collaborative Memory, Zep, SemDeDup were set from **full-text** reads. Others use the Phase-1 matrix and should be re-confirmed from full text before the camera-ready related-work section.
