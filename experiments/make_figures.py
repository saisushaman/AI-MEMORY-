"""Generate paper figures from the real measured results. No LLM calls.
Outputs PNGs into paper/figures/. All numbers are the measured values reported
in research/phase3_pilot_results.md and research/pivot_diagnostic_results.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ---- Fig 1: budget sweep (LLM-judged accuracy vs k) --------------------------
k = [2, 4, 8, 12]
full = [0.265, 0.330, 0.375, 0.400]
naive = [0.150, 0.180, 0.205, 0.210]
qaware = [0.240, 0.285, 0.370, 0.400]
plt.figure(figsize=(5, 3.4))
plt.plot(k, full, "o-", label="full (no dedup)")
plt.plot(k, qaware, "s--", label="query-aware (MMR)")
plt.plot(k, naive, "^:", label="naive-dedup")
plt.xlabel("retrieval budget $k$ (facts)")
plt.ylabel("LLM-judged accuracy")
plt.title("Deduplication does not help (LoCoMo, cats 1--4, $n{=}200$)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig_budget_sweep.png"); plt.close()

# ---- Fig 2: oracle bottleneck by category ------------------------------------
cats = ["overall", "cat1\nmulti-hop", "cat2\ntemporal", "cat3\nopen", "cat4\nsingle"]
oracle = [0.610, 0.721, 0.156, 0.368, 0.828]
fullconv = [0.630, 0.628, 0.244, 0.421, 0.860]
retrieved = [0.400, 0.411, np.nan, np.nan, 0.613]
x = np.arange(len(cats)); w = 0.27
plt.figure(figsize=(6.2, 3.6))
plt.bar(x - w, oracle, w, label="oracle (gold evidence)")
plt.bar(x, fullconv, w, label="full conversation")
plt.bar(x + w, retrieved, w, label="retrieved (top-$k$)")
plt.xticks(x, cats); plt.ylabel("LLM-judged accuracy")
plt.title("Bottleneck: retrieval precision, then reasoning ($n{=}200$)")
plt.legend(fontsize=9); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig_oracle.png"); plt.close()

# ---- Fig 3: redundancy is semantic not lexical -------------------------------
tau = [0.95, 0.90, 0.85, 0.80, 0.75]
naive_store = [0.06, 0.29, 0.62, 0.87, 0.96]
gated_store = [0.00, 0.00, 0.00, 0.64, 0.92]
plt.figure(figsize=(5, 3.4))
plt.plot(tau, naive_store, "o-", label="naive store")
plt.plot(tau, gated_store, "s--", label="gated@0.85 store")
plt.axhline(0.01, color="gray", ls=":", label="lexical (all $\\tau$)")
plt.gca().invert_xaxis()
plt.xlabel("semantic similarity threshold $\\tau$")
plt.ylabel("Memory Duplication Ratio (MDR)")
plt.title("Redundancy is semantic, invisible to lexical matching")
plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig_redundancy.png"); plt.close()

# ---- Fig 4: LongMemEval oracle accuracy by question type ---------------------
types = ["single-sess\nassistant", "single-sess\nuser", "knowledge\nupdate",
         "single-sess\npreference", "multi-\nsession", "temporal\nreasoning"]
acc = [0.950, 0.800, 0.600, 0.467, 0.375, 0.275]
colors = ["#4C72B0"]*4 + ["#DD8452", "#C44E52"]
plt.figure(figsize=(6.2, 3.4))
plt.bar(range(len(types)), acc, color=colors)
plt.xticks(range(len(types)), types, fontsize=8)
plt.ylabel("oracle accuracy (gold evidence)")
plt.title("LongMemEval: temporal reasoning is the floor, even with gold evidence")
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/fig_longmemeval.png"); plt.close()

print("wrote:", os.listdir(OUT))
