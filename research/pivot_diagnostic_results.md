# Pivot Diagnostic — Where the Accuracy Actually Leaks (2026-07-30)

After the dedup direction was empirically rejected (see [`phase3_pilot_results.md`](phase3_pilot_results.md)), we pivoted to the bottleneck the data exposed. This diagnostic uses LoCoMo's **gold evidence turns** (dia_ids) to measure the *ceiling* that retrieval could reach. Script: [`../experiments/oracle_diagnostic.py`](../experiments/oracle_diagnostic.py); output `../experiments/oracle_diagnostic_output.txt`. LLM-judged, n=200 (cats 1–4), local qwen3:8b, `num_ctx=32768` so the full-conversation arm is not truncated.

Arms:
- **oracle** — context = only the gold evidence turns (with session dates) → perfect-retrieval ceiling
- **full_conv** — context = the entire multi-session conversation (no retrieval) → long-context ceiling
- **retrieved** — our earlier RAG pipeline (top-k facts by embedding sim)

## Results

| arm | overall | cat1 multi-hop | cat2 temporal | cat3 open | cat4 single |
|---|--:|--:|--:|--:|--:|
| oracle | 0.610 | **0.721** | 0.156 | 0.368 | 0.828 |
| full_conv | 0.630 | 0.628 | 0.244 | 0.421 | 0.860 |
| retrieved (full@k12) | 0.400 | ~0.411 | — | — | ~0.61 |

## Findings

1. **Retrieval precision is a large, measured bottleneck.** Oracle 0.61 vs our retrieval 0.40 = ~21 points lost to imperfect retrieval; on **multi-hop it is 0.41 → 0.72 (a 31-point gap)**. This is real headroom for a method (contrast: the dedup direction had none).

2. **Precision beats recall.** On multi-hop, full_conv (0.628) is *worse* than the precise oracle evidence (0.721) — adding the whole conversation **distracts** the model. The lever is "retrieve exactly the right multi-hop evidence chain," not "retrieve more."

3. **Temporal (cat2) is a different bottleneck retrieval cannot fix.** Oracle is only 0.156 — even perfect evidence + session dates don't help because the model can't do the date arithmetic ("yesterday" + session date → calendar date). This is a reasoning/grounding problem; correctly out of scope for a retrieval method.

4. **Single-hop (cat4)** oracle 0.828 — high; retrieval headroom (~0.61 → 0.83) exists but the harder, more distinctive win is multi-hop.

## Data-grounded method target

**Precision-oriented multi-hop evidence retrieval over conversational agent memory** — assemble the exact multi-hop evidence chain (close the 0.41 → 0.72 gap), exploiting that *less-but-right* beats *more*. Temporal grounding is a separate, acknowledged limitation.

## Caveats / honesty
- Single dataset (LoCoMo), single model (qwen3:8b), n=200; oracle uses LoCoMo's own evidence annotations (imperfect but standard).
- "retrieved" numbers are from the earlier fact-store RAG pipeline (nomic-embed); a like-for-like retrieval baseline will be re-run in the method phase.
- The oracle ceiling (0.61) is itself below 1.0 — some questions are unanswerable even with gold evidence (reasoning/reformulation), bounding how much *any* retrieval method can gain.

## Next step (discipline carried over from the dedup post-mortem)
**Novelty check BEFORE building.** Multi-hop / iterative retrieval and query decomposition are well-studied (IRCoT, self-ask, HippoRAG, etc.). Before committing to a method, verify what is genuinely novel about *precision-oriented multi-hop retrieval over multi-session conversational memory* specifically — do not assume novelty (that error is what sent the dedup direction into a dead end). Then design + pilot the method against the oracle ceiling.
