"""
Phase-2 go/no-go pilot: does semantic deduplication of a shared canonical-unit
store reduce BOTH storage duplication AND per-query context tokens, and is there
a regime where naive dedup hurts task utility?

This is a SYNTHETIC pilot, not a benchmark result. Its only job is to tell us
whether the "coupling" conjecture behind Direction A is non-trivial and worth a
theorem, before we invest in the full system. All numbers are simulation output
under a fixed seed and are clearly labelled as synthetic.

Model
-----
- K latent facts. A query "needs" a random subset of facts (and sometimes a
  query-relevant *distinction* between near-duplicate units).
- Store: each fact is backed by R near-duplicate units (redundancy). Units for
  the same fact are semantic duplicates EXCEPT a fraction rho of duplicate
  clusters that actually carry a task-relevant distinction (merging them loses
  info some queries need).
- Composers:
    oracle : optimal min-token set cover of the needed facts (implicit dedup).
    topk   : realistic retriever - take top-k units by (relevance + noise),
             fill context up to budget; near-dups score alike so top-k pulls
             multiple copies of the same fact => wasted tokens.
- Dedup strategies:
    none        : keep all units.
    naive       : one canonical unit per semantic cluster (drops distinctions).
    query_aware : merge a cluster only if its internal distinction is never
                  needed by the query distribution (keeps distinctions that matter).
- Metrics: stored units, stored tokens, mean per-query context tokens, task
  success rate (needed facts + any needed distinction all covered).
"""

import numpy as np

SEED = 20260721
rng = np.random.default_rng(SEED)

# ---- config ---------------------------------------------------------------
K_FACTS = 200          # latent facts
R_DUP = 4              # near-duplicate units per fact (redundancy level)
TOK_MIN, TOK_MAX = 20, 60   # per-unit token cost range
N_QUERIES = 4000
FACTS_PER_QUERY = 5    # facts a query needs
TOPK = 12              # retriever cardinality
BUDGET = 300           # per-query token budget
RHO_GRID = [0.0, 0.1, 0.3]  # fraction of clusters with a task-relevant distinction

# ---- build the store ------------------------------------------------------
# unit fields: fact_id, variant_id (0=base,1=distinct), token cost
def build_store():
    units = []
    cluster_has_distinction = rng.random(K_FACTS)  # thresholded per rho later
    for f in range(K_FACTS):
        for r in range(R_DUP):
            # variant 1 (the "distinct" one) is a single designated member
            variant = 1 if r == R_DUP - 1 else 0
            tok = int(rng.integers(TOK_MIN, TOK_MAX + 1))
            units.append({"fact": f, "variant": variant, "tok": tok})
    return units, cluster_has_distinction

def make_queries(cluster_has_distinction, rho):
    distinct_clusters = set(np.where(cluster_has_distinction < rho)[0].tolist())
    queries = []
    for _ in range(N_QUERIES):
        needed = rng.choice(K_FACTS, size=FACTS_PER_QUERY, replace=False).tolist()
        # if a needed fact's cluster carries a distinction, the query needs the
        # distinct variant with prob 0.5 (else the base fact suffices)
        need_distinct = {}
        for f in needed:
            if f in distinct_clusters and rng.random() < 0.5:
                need_distinct[f] = 1
            else:
                need_distinct[f] = 0
        queries.append((needed, need_distinct))
    return queries, distinct_clusters

# ---- dedup strategies -> produce the retained unit set --------------------
def dedup(units, strategy, distinct_clusters):
    if strategy == "none":
        return units
    retained = []
    # group by fact
    by_fact = {}
    for u in units:
        by_fact.setdefault(u["fact"], []).append(u)
    for f, group in by_fact.items():
        keep_distinction = (strategy == "query_aware") and (f in distinct_clusters)
        if keep_distinction:
            # keep cheapest base + the distinct variant
            base = [u for u in group if u["variant"] == 0]
            dist = [u for u in group if u["variant"] == 1]
            if base:
                retained.append(min(base, key=lambda u: u["tok"]))
            if dist:
                retained.append(min(dist, key=lambda u: u["tok"]))
        else:
            # naive OR query_aware-without-needed-distinction: one canonical
            retained.append(min(group, key=lambda u: u["tok"]))
    return retained

# ---- composers ------------------------------------------------------------
def compose_oracle(units, needed, need_distinct):
    # optimal per-fact: cheapest unit satisfying the (fact, needed-variant)
    ctx = []
    for f in needed:
        cands = [u for u in units if u["fact"] == f]
        if need_distinct.get(f, 0) == 1:
            cands = [u for u in cands if u["variant"] == 1]
        if not cands:
            continue
        ctx.append(min(cands, key=lambda u: u["tok"]))
    return ctx

def compose_topk(units, needed, need_distinct):
    needed_set = set(needed)
    # relevance = 1 if unit's fact is needed, else 0; + gaussian noise
    scored = []
    for u in units:
        rel = 1.0 if u["fact"] in needed_set else 0.0
        score = rel + rng.normal(0, 0.35)
        scored.append((score, u))
    scored.sort(key=lambda x: -x[0])
    ctx, toks = [], 0
    for _, u in scored[:TOPK]:
        if toks + u["tok"] > BUDGET:
            continue
        ctx.append(u)
        toks += u["tok"]
    return ctx

def success(ctx, needed, need_distinct):
    covered = {}
    for u in ctx:
        covered.setdefault(u["fact"], set()).add(u["variant"])
    for f in needed:
        if f not in covered:
            return 0
        if need_distinct.get(f, 0) == 1 and 1 not in covered[f]:
            return 0
    return 1

# ---- run ------------------------------------------------------------------
def evaluate(units, queries, distinct_clusters, composer):
    tot_tok, tot_succ = 0, 0
    fn = compose_oracle if composer == "oracle" else compose_topk
    for needed, need_distinct in queries:
        ctx = fn(units, needed, need_distinct)
        tot_tok += sum(u["tok"] for u in ctx)
        tot_succ += success(ctx, needed, need_distinct)
    n = len(queries)
    return tot_tok / n, tot_succ / n

def store_stats(units):
    return len(units), sum(u["tok"] for u in units)

base_units, cluster_flags = build_store()
full_units_n, full_tok = store_stats(base_units)

print("=" * 78)
print("SYNTHETIC COUPLING PILOT  (seed=%d) -- NOT a benchmark result" % SEED)
print("=" * 78)
print(f"Full store: {full_units_n} units, {full_tok} stored tokens "
      f"(K={K_FACTS} facts x R={R_DUP} near-dups)")
print()

for rho in RHO_GRID:
    queries, distinct_clusters = make_queries(cluster_flags, rho)
    print(f"--- rho = {rho:.2f}  (fraction of clusters with a task-relevant distinction) ---")
    header = f"{'dedup':<12}{'composer':<9}{'units':>7}{'stored_tok':>12}{'ctx_tok/q':>11}{'success':>9}"
    print(header)
    for strategy in ["none", "naive", "query_aware"]:
        retained = dedup(base_units, strategy, distinct_clusters)
        u_n, s_tok = store_stats(retained)
        for composer in ["oracle", "topk"]:
            ctx_tok, succ = evaluate(retained, queries, distinct_clusters, composer)
            print(f"{strategy:<12}{composer:<9}{u_n:>7}{s_tok:>12}{ctx_tok:>11.1f}{succ:>9.3f}")
    print()

print("Read: compare 'ctx_tok/q' between composers (oracle=optimal, topk=realistic)")
print("and 'success' between dedup=naive vs query_aware as rho grows.")
