"""Watch the second-answerer cache; when it stops growing (or completes),
compute and print the accuracy table from the cache. Read-only."""
import json, os, time
CACHE = "experiments/cache"
jf = os.path.join(CACHE, "qa_judge_llama32ans.json")
TARGET = 200 * 3 * 2  # ~1200 (n=200 x 3 policies x 2 budgets), minus any skipped

def count():
    return len(json.load(open(jf, encoding="utf-8"))) if os.path.exists(jf) else 0

stable = 0
prev = -1
for _ in range(400):  # up to ~200 min
    n = count()
    if n >= TARGET or (n == prev and n > 0):
        stable += 1
    else:
        stable = 0
    prev = n
    if stable >= 3:
        break
    time.sleep(30)

# aggregate
jud = json.load(open(jf, encoding="utf-8")) if os.path.exists(jf) else {}
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
for key, ok in jud.items():
    parts = key.split("|"); pol = parts[-2]; k = parts[-1]
    agg[(pol, k)][0] += 1 if ok else 0
    agg[(pol, k)][1] += 1
KS = ["8", "12"]; POLS = ["full", "naive_dedup", "query_aware"]
print("=" * 60)
print("SECOND ANSWERER = llama3.2:3b (judge=qwen3:8b, LoCoMo cats1-4)")
print("=" * 60)
print(f"{'policy':<13}" + "".join(f"k={k:>2}   " for k in KS))
for pol in POLS:
    row = f"{pol:<13}"
    for k in KS:
        c, tot = agg[(pol, k)]; row += f"{(c/tot if tot else 0):>6.3f} "
    n0 = agg[(POLS[0], KS[0])][1]
    print(row)
print(f"(n per cell ~ {agg[(POLS[0],KS[0])][1]}; total judged {len(jud)})")
print("pattern check: naive_dedup dominated? query_aware ~ full?")
