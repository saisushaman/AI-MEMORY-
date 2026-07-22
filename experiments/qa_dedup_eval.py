"""
Phase-3c: does deduplication affect ANSWER QUALITY, not just storage? The
"without degrading task performance" half of the research question.

Three memory policies answer LoCoMo QA (local qwen3:8b over a fact store built
from cached Phase-3b extractions; retrieval by nomic-embed cosine):

  full        : retrieve top-k facts from the FULL store (no dedup)
  naive_dedup : blindly canonicalize the store at TAU_DEDUP (merge near-dups,
                keep shortest), then retrieve top-k  -> tests over-merge harm
  query_aware : full store; select k facts by MMR = relevance - LAMBDA*redundancy
                -> keeps query-relevant distinctions, drops redundant ones

Metrics per policy: token-F1 vs gold, containment accuracy, mean context tokens.
Honest notes: MMR is a classic composer (proxy for the query-aware idea; the full
store-level distribution-aware method is future work). Automatic F1/containment
(no LLM-judge) is the primary metric for reproducibility; single dataset/model.
Resumable: answers cached per (policy, conv, qidx).
"""
import urllib.request, json, os, re, time, random
import numpy as np

OLLAMA="http://localhost:11434"; GEN="qwen3:8b"; EMB="nomic-embed-text"
DATA="experiments/data/locomo10.json"; CACHE="experiments/cache"
K=8                      # facts injected per question
TAU_DEDUP=0.82           # naive store-dedup threshold (mid of the near-dup band from Phase 3b)
LAMBDA=1.0               # MMR redundancy weight
CAND=30                  # candidate pool for MMR
QPER=20                  # questions sampled per conversation
CATS={"1","2","3","4"}   # skip cat 5 (adversarial/unanswerable) for the accuracy pilot
SEED=7

def _post(path,body,timeout):
    r=urllib.request.urlopen(urllib.request.Request(OLLAMA+path,
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=timeout)
    return json.loads(r.read())
def embed(text): return _post("/api/embeddings",{"model":EMB,"prompt":text},120)["embedding"]
def answer(context,q):
    prompt=("Answer the question using ONLY the memory facts. Be concise; reply with just "
            "the answer, no explanation. If unknown, say 'unknown'.\n\nMemory facts:\n"
            f"{context}\n\nQuestion: {q}\nAnswer:")
    out=_post("/api/chat",{"model":GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"options":{"temperature":0,"seed":0}},600)
    return out["message"]["content"].strip()
def tok(t): return max(1,len(t.split()))

_word=re.compile(r"[a-z0-9]+")
def norm(t): return _word.findall(str(t).lower())
def f1(pred,gold):
    p,g=norm(pred),norm(gold)
    if not p or not g: return 0.0
    from collections import Counter
    common=Counter(p)&Counter(g); ncommon=sum(common.values())
    if ncommon==0: return 0.0
    prec=ncommon/len(p); rec=ncommon/len(g)
    return 2*prec*rec/(prec+rec)
def contains(pred,gold):
    return 1.0 if " ".join(norm(gold)) in " ".join(norm(pred)) else 0.0

def conv_store(ex_conv):
    seen=set(); facts=[]
    for sess in ex_conv:
        for f in sess:
            if f not in seen: seen.add(f); facts.append(f)
    return facts

def dedup_store(facts,E,tau):
    n=len(facts); sim=E@E.T; parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    iu=np.triu_indices(n,1)
    for p in np.where(sim[iu]>=tau)[0]:
        a,b=int(iu[0][p]),int(iu[1][p]); ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(i)
    keep=[min(g,key=lambda i:tok(facts[i])) for g in groups.values()]  # canonical=shortest
    keep.sort()
    return keep

def topk_by_relevance(qv,E,idxs,k):
    sims=E[idxs]@qv
    order=np.argsort(-sims)[:k]
    return [idxs[i] for i in order]

def mmr(qv,E,idxs,k,lam):
    cand=topk_by_relevance(qv,E,idxs,min(CAND,len(idxs)))
    selected=[]
    while cand and len(selected)<k:
        best,bi=None,None
        for c in cand:
            rel=float(E[c]@qv)
            red=max((float(E[c]@E[s]) for s in selected),default=0.0)
            score=rel-lam*red
            if best is None or score>best: best,bi=score,c
        selected.append(bi); cand.remove(bi)
    return selected

def main():
    data=json.load(open(DATA,encoding="utf-8"))
    ex=json.load(open(os.path.join(CACHE,"extractions.json"),encoding="utf-8"))
    embmap=json.load(open(os.path.join(CACHE,"embeddings.json"),encoding="utf-8"))
    qcache_f=os.path.join(CACHE,"q_embeddings.json")
    qcache=json.load(open(qcache_f,encoding="utf-8")) if os.path.exists(qcache_f) else {}
    acache_f=os.path.join(CACHE,"qa_answers.json")
    acache=json.load(open(acache_f,encoding="utf-8")) if os.path.exists(acache_f) else {}
    rng=random.Random(SEED)

    policies=["full","naive_dedup","query_aware"]
    agg={p:{"f1":0.0,"contain":0.0,"ctx_tok":0,"n":0} for p in policies}
    t0=time.time(); done=0
    for ci,sample in enumerate(data):
        facts=conv_store(ex[ci])
        E=np.asarray([embmap[f] for f in facts],dtype=np.float32)
        E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
        dd_idx=dedup_store(facts,E,TAU_DEDUP)     # naive-dedup canonical index set
        full_idx=list(range(len(facts)))
        qs=[q for q in sample["qa"] if str(q.get("category")) in CATS and q.get("answer")]
        rng.shuffle(qs); qs=qs[:QPER]
        for qi,q in enumerate(qs):
            question=q["question"]; gold=str(q["answer"])
            if question not in qcache: qcache[question]=embed(question)
            qv=np.asarray(qcache[question],dtype=np.float32); qv/=np.linalg.norm(qv)+1e-9
            for pol in policies:
                if pol=="full": sel=topk_by_relevance(qv,E,full_idx,K)
                elif pol=="naive_dedup": sel=topk_by_relevance(qv,E,dd_idx,K)
                else: sel=mmr(qv,E,full_idx,K,LAMBDA)
                ctx="\n".join(f"- {facts[i]}" for i in sel)
                ctx_tok=sum(tok(facts[i]) for i in sel)
                akey=f"{ci}|{q['question']}|{pol}"
                if akey not in acache:
                    acache[akey]=answer(ctx,question)
                    done+=1
                    if done%50==0:
                        json.dump(acache,open(acache_f,"w",encoding="utf-8"))
                        json.dump(qcache,open(qcache_f,"w",encoding="utf-8"))
                        print(f"  {done} answers t={time.time()-t0:.0f}s",flush=True)
                pred=acache[akey]
                a=agg[pol]; a["f1"]+=f1(pred,gold); a["contain"]+=contains(pred,gold)
                a["ctx_tok"]+=ctx_tok; a["n"]+=1
        print(f"conv {ci+1}/10 done ({len(qs)} qs) t={time.time()-t0:.0f}s",flush=True)
    json.dump(acache,open(acache_f,"w",encoding="utf-8"))
    json.dump(qcache,open(qcache_f,"w",encoding="utf-8"))

    print("\n"+"="*76)
    print("QA UNDER DEDUP POLICIES (LoCoMo cats 1-4; qwen3:8b; k=%d facts)"%K)
    print("store-dedup tau=%.2f | MMR lambda=%.1f | %d questions/conv"%(TAU_DEDUP,LAMBDA,QPER))
    print("="*76)
    print(f"{'policy':<14}{'n':>5}{'F1':>8}{'contain':>9}{'ctx_tok/q':>11}")
    for p in policies:
        a=agg[p]; n=max(1,a["n"])
        print(f"{p:<14}{a['n']:>5}{a['f1']/n:>8.3f}{a['contain']/n:>9.3f}{a['ctx_tok']/n:>11.1f}")
    print("\nRead: does naive_dedup lose F1/containment vs full (over-merge harm)?")
    print("Does query_aware match/beat full at fewer ctx tokens (efficiency without quality loss)?")

if __name__=="__main__":
    main()
