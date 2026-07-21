"""
Phase-3 real-data pilot (Tier 2, semantic): measure PARAPHRASE-LEVEL duplication
in the LoCoMo extracted-`observation` memory stream using sentence embeddings.

Tier 1 (lexical Jaccard) found near-zero duplication -> if duplication exists it
is semantic. This script tests that with all-MiniLM-L6-v2 cosine similarity.

Same metrics as Tier 1:
  MDR = 1 - (#canonical / #units)      (fraction of units that are semantic near-dups)
  RTF = (total_tokens - canonical_tokens)/total_tokens
Near-duplicate = cosine >= TAU, greedy single-link clustering.

Honest framing: this measures redundancy in LoCoMo's extracted-observation stream
(a faithful analog to memory units), NOT the internal store of a specific system.
Thresholds are reported across a range because "semantic duplicate" is threshold-
dependent; we show the whole curve rather than cherry-picking one.
"""
import json, re
import numpy as np
from sentence_transformers import SentenceTransformer

DATA = "experiments/data/locomo10.json"
TAUS = [0.95, 0.90, 0.85, 0.80, 0.75]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def tok_count(t): return max(1, len(t.split()))

def collect_observation_facts(sample):
    facts = []
    for sess_key, per_speaker in sample.get("observation", {}).items():
        if not isinstance(per_speaker, dict): continue
        for speaker, flist in per_speaker.items():
            if not isinstance(flist, list): continue
            for item in flist:
                if isinstance(item, list) and item: facts.append(str(item[0]))
                elif isinstance(item, str): facts.append(item)
    return facts

def cluster_cosine(emb, tau):
    """Greedy single-link clustering by cosine>=tau. emb: (n,d) L2-normalized."""
    n = emb.shape[0]
    sim = emb @ emb.T
    parent = list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y):
        rx,ry=find(x),find(y)
        if rx!=ry: parent[rx]=ry
    iu = np.triu_indices(n,1)
    pairs = np.where(sim[iu] >= tau)[0]
    for p in pairs:
        union(int(iu[0][p]), int(iu[1][p]))
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(i)
    return list(groups.values())

def analyze(strings, emb):
    total=len(strings); total_tokens=sum(tok_count(s) for s in strings)
    out={}
    for tau in TAUS:
        cl=cluster_cosine(emb,tau)
        n_canon=len(cl)
        canon_tokens=sum(min(tok_count(strings[i]) for i in c) for c in cl)
        mdr=1-n_canon/total if total else 0
        rtf=(total_tokens-canon_tokens)/total_tokens if total_tokens else 0
        mx=max((len(c) for c in cl),default=0)
        # count clusters with >=2 members (actual duplicate groups)
        dup_groups=sum(1 for c in cl if len(c)>=2)
        out[tau]=(total,n_canon,mdr,rtf,mx,dup_groups)
    return out

def main():
    data=json.load(open(DATA,encoding="utf-8"))
    print("Loading model:", MODEL)
    model=SentenceTransformer(MODEL)
    print("="*100)
    print("LoCoMo REAL-DATA duplication pilot (Tier 2: SEMANTIC cosine, all-MiniLM-L6-v2) -- `observation` facts")
    print("="*100)

    # embed everything once
    per_sample=[collect_observation_facts(s) for s in data]
    all_strings=[x for p in per_sample for x in p]
    print(f"Embedding {len(all_strings)} observation facts...")
    all_emb=model.encode(all_strings, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    all_emb=np.asarray(all_emb,dtype=np.float32)

    # per-conversation (within one agent-pair's own multi-session history)
    print(f"\n### per-conversation average (within a single conversation's multi-session history)")
    print(f"{'tau':>6}{'units':>8}{'canonical':>11}{'MDR':>8}{'RTF':>8}{'dup_grps':>9}{'max_dup':>9}")
    agg={tau:[0,0,0.0,0.0,0,0] for tau in TAUS}; n=0; off=0
    for p in per_sample:
        if not p: continue
        n+=1
        e=all_emb[off:off+len(p)]; off+=len(p)
        row=analyze(p,e)
        for tau in TAUS:
            t,c,m,r,mx,dg=row[tau]
            a=agg[tau]; a[0]+=t;a[1]+=c;a[2]+=m;a[3]+=r;a[4]=max(a[4],mx);a[5]+=dg
    for tau in TAUS:
        t,c,m,r,mx,dg=agg[tau]
        print(f"{tau:>6.2f}{t//n:>8}{c//n:>11}{m/n:>8.3f}{r/n:>8.3f}{dg//n:>9}{mx:>9}")

    # corpus-wide (pool all conversations -> cross-agent duplication, e.g. shared world facts)
    print(f"\n### corpus-wide (all {len(all_strings)} facts pooled across 10 conversations)")
    print(f"{'tau':>6}{'units':>8}{'canonical':>11}{'MDR':>8}{'RTF':>8}{'dup_grps':>9}{'max_dup':>9}")
    row=analyze(all_strings, all_emb)
    for tau in TAUS:
        t,c,m,r,mx,dg=row[tau]
        print(f"{tau:>6.2f}{t:>8}{c:>11}{m:>8.3f}{r:>8.3f}{dg:>9}{mx:>9}")

    # show a few example near-duplicate pairs at tau=0.85 (evidence, not cherry-picked metric)
    print("\n### example semantic near-duplicate pairs (corpus, cosine>=0.85):")
    sim=all_emb@all_emb.T
    iu=np.triu_indices(len(all_strings),1)
    idx=np.where(sim[iu]>=0.85)[0]
    shown=0
    for p in idx:
        i,j=int(iu[0][p]),int(iu[1][p])
        print(f"  [{sim[i,j]:.2f}] {all_strings[i][:70]!r}  <=>  {all_strings[j][:70]!r}")
        shown+=1
        if shown>=8: break
    if shown==0: print("  (none at 0.85)")

if __name__=="__main__":
    main()
