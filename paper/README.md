# Paper

**An Empirical Anatomy of Failure in Small-Model Long-Term Conversational Memory: Redundancy, Retrieval, and Reasoning**

A workshop / short empirical-analysis paper built entirely from the real, reproducible experiments in [`../experiments/`](../experiments/). **Every number is measured** (no fabricated results); see `main.tex` header for the honesty policy.

## Compile

Self-contained — compiles on **Overleaf** (or locally) with `pdflatex` + `bibtex`, no external style files:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

To target a specific venue, replace `\documentclass` and the title block in `main.tex` with that venue's style (e.g. `acl.sty`, `neurips_2026.sty`); the `sections/*.tex` are reusable as-is.

## Structure

```
main.tex            entry point (\input's the sections)
references.bib      bibliography (all entries verified real)
sections/           abstract, introduction, related_work, setup, redundancy,
                    dedup, bottleneck, faithfulness, discussion, limitations,
                    conclusion
tables/  figures/   (numbers are currently inline in sections)
```

## Honest positioning

This is an **empirical analysis / negative-result** paper (workshop / arXiv tier), **not** a novel-method paper. The method space (deduplication, precision conversational-memory retrieval, memory-hallucination mitigation) is already covered by very recent work, which the Related Work section cites directly. The contribution is a controlled, reproducible decomposition of *where* a small local agent-memory pipeline loses accuracy. See [`../research/`](../research/) for the full novelty analyses that led here.
