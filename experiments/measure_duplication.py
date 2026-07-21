"""
Phase-3 real-data pilot (Tier 1, dependency-free): measure redundancy in the
LoCoMo *extracted-memory* stream.

We use LoCoMo's own `observation` field: per session, per speaker, a list of
atomic factual statements [fact_text, dia_id]. These are the closest public
analog to what an agent memory system would store as units. We measure how much
of that memory stream is duplicated ACROSS the multi-session history.

This is REAL data but a LEXICAL proxy for semantic duplication (Tier 2 adds
embeddings). It is honest about what it measures: redundancy in the extracted-
observation stream, not the internal store of any specific system (that needs
running Mem0/A-MEM, which needs an LLM API).

Metrics (defined here, used across the project):
  Memory Duplication Ratio  MDR = 1 - (#canonical units / #total units)
  Redundant Token Fraction  RTF = (total_tokens - canonical_tokens) / total_tokens
    where a "canonical unit" is one near-duplicate cluster and canonical_tokens
    keeps the shortest member of each cluster.
Near-duplicate = normalized-token Jaccard >= TAU (greedy single-link clustering).
"""
import json, re, sys

DATA = "experiments/data/locomo10.json"
TAUS = [1.00, 0.80, 0.60]   # 1.00 == exact (normalized) duplicates
STOP = set("a an the is are was were be been being to of in on at for and or "
           "with her his their my your our its it that this these those as by "
           "i you he she they we".split())

def norm_tokens(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    toks = [w for w in t.split() if w not in STOP]
    return toks

def jaccard(a, b):
    if not a or not b: return 0.0
    A, B = set(a), set(b)
    return len(A & B) / len(A | B)

def cluster(items, tau):
    """Greedy single-link clustering by Jaccard >= tau. items: list of token lists.
    Returns list of clusters (each a list of indices)."""
    n = len(items)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    # blocking by shared token to cut comparisons
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, toks in enumerate(items):
        for w in set(toks):
            buckets[w].append(i)
    seen = set()
    for w, idxs in buckets.items():
        if len(idxs) < 2: continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                key = (i, j)
                if key in seen: continue
                seen.add(key)
                if jaccard(items[i], items[j]) >= tau:
                    union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())

def tok_count(t):
    return max(1, len(t.split()))

def collect_observation_facts(sample):
    facts = []  # (text)
    obs = sample.get("observation", {})
    for sess_key, per_speaker in obs.items():
        if not isinstance(per_speaker, dict): continue
        for speaker, flist in per_speaker.items():
            if not isinstance(flist, list): continue
            for item in flist:
                if isinstance(item, list) and item:
                    facts.append(str(item[0]))
                elif isinstance(item, str):
                    facts.append(item)
    return facts

def collect_turns(sample):
    conv = sample.get("conversation", {})
    turns = []
    for k, v in conv.items():
        if isinstance(v, list):
            for t in v:
                if isinstance(t, dict) and "text" in t:
                    turns.append(str(t["text"]))
    return turns

def analyze(strings):
    toks = [norm_tokens(s) for s in strings]
    total = len(strings)
    total_tokens = sum(tok_count(s) for s in strings)
    row = {}
    for tau in TAUS:
        clusters = cluster(toks, tau)
        n_canon = len(clusters)
        # redundant tokens: keep shortest member per cluster
        canon_tokens = 0
        for cl in clusters:
            canon_tokens += min(tok_count(strings[i]) for i in cl)
        mdr = 1 - n_canon / total if total else 0.0
        rtf = (total_tokens - canon_tokens) / total_tokens if total_tokens else 0.0
        # size of largest cluster (how many times the most-repeated fact recurs)
        max_cl = max((len(cl) for cl in clusters), default=0)
        row[tau] = (total, n_canon, mdr, rtf, max_cl)
    return row, total, total_tokens

def main():
    data = json.load(open(DATA, encoding="utf-8"))
    print("=" * 92)
    print("LoCoMo REAL-DATA duplication pilot (Tier 1: lexical Jaccard) -- extracted `observation` facts")
    print("=" * 92)
    for field, collect in [("observation-facts", collect_observation_facts),
                           ("raw-turns", collect_turns)]:
        # corpus-wide (pool all 10 samples) AND per-sample average
        all_strings = []
        per_sample = []
        for s in data:
            strings = collect(s)
            all_strings.extend(strings)
            per_sample.append(strings)
        print(f"\n### field = {field}  (corpus: {len(all_strings)} units across {len(data)} conversations)")
        print(f"{'scope':<16}{'tau':>6}{'units':>8}{'canonical':>11}{'MDR':>8}{'RTF':>8}{'max_dup':>9}")
        # per-sample averaged (duplication WITHIN a single agent's own history)
        agg = {tau: [0,0,0.0,0.0,0] for tau in TAUS}
        for strings in per_sample:
            if not strings: continue
            row, _, _ = analyze(strings)
            for tau in TAUS:
                t,c,m,r,mx = row[tau]
                agg[tau][0]+=t; agg[tau][1]+=c; agg[tau][2]+=m; agg[tau][3]+=r; agg[tau][4]=max(agg[tau][4],mx)
        n=len([p for p in per_sample if p])
        for tau in TAUS:
            t,c,m,r,mx=agg[tau]
            print(f"{'per-conv avg':<16}{tau:>6.2f}{t//n:>8}{c//n:>11}{m/n:>8.3f}{r/n:>8.3f}{mx:>9}")
    print("\nMDR = 1 - canonical/units (fraction of memory units that are near-duplicates of another)")
    print("RTF = fraction of stored tokens attributable to redundant copies")
    print("max_dup = largest near-duplicate cluster (times a single fact recurs)")
    print("\nNOTE: lexical proxy; Tier 2 (embeddings) will catch paraphrase-level duplication this misses (so these are LOWER bounds).")

if __name__ == "__main__":
    main()
