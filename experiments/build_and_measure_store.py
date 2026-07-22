"""
Phase-3b: build a REAL memory store from LoCoMo with a local LLM (Ollama qwen3:8b)
the way Mem0 / A-MEM / LangMem do (extract atomic facts per session, accumulate),
then measure duplication OF THAT STORE. This upgrades the claim from "the
observation stream is redundant" to "an LLM-extracted memory store is N% duplicate".

Two store policies:
  naive : add every extracted fact (what a naive extract-then-store pipeline does)
  gated : Mem0-style ADD/NOOP gate -- before adding, embed and compare to the
          existing store; skip if max cosine >= GATE (a similarity-gated dedup)

Metrics (same as the rest of the project):
  MDR = 1 - canonical/units ,  RTF = redundant-token fraction  (semantic cosine)

Local only: Ollama at localhost:11434 (qwen3:8b + nomic-embed-text). No API key.
Extractions and embeddings are cached to experiments/cache/ for reproducibility.
"""
import urllib.request, json, os, re, time
import numpy as np

OLLAMA="http://localhost:11434"
GEN="qwen3:8b"; EMB="nomic-embed-text"
DATA="experiments/data/locomo10.json"
CACHE="experiments/cache"; os.makedirs(CACHE,exist_ok=True)
GATE=0.85                      # Mem0-style dedup gate
TAUS=[0.95,0.90,0.85,0.80,0.75]

def _post(path,body,timeout):
    r=urllib.request.urlopen(urllib.request.Request(OLLAMA+path,
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=timeout)
    return json.loads(r.read())

def extract_facts(session_text):
    prompt=("You are a memory extractor. From the dialogue below, extract atomic, "
            "self-contained factual memories about the speakers (one fact each, "
            "third person, no pronouns-only). Return JSON: {\"facts\":[\"...\",...]}.\n\n"
            f"Dialogue:\n{session_text}")
    out=_post("/api/chat",{"model":GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"format":"json","options":{"temperature":0,"seed":0}},600)
    try:
        obj=json.loads(out["message"]["content"])
        facts=obj.get("facts",[])
        return [str(f).strip() for f in facts if isinstance(f,(str,)) and str(f).strip()]
    except Exception:
        return []

def embed(text):
    return _post("/api/embeddings",{"model":EMB,"prompt":text},120)["embedding"]

def tok(t): return max(1,len(t.split()))

def session_texts(sample):
    conv=sample.get("conversation",{}); sessions=[]
    for k in sorted([k for k in conv if k.startswith("session_") and isinstance(conv[k],list)],
                    key=lambda x:int(re.findall(r"\d+",x)[0])):
        lines=[f"{t.get('speaker','?')}: {t.get('text','')}" for t in conv[k] if isinstance(t,dict)]
        sessions.append("\n".join(lines))
    return sessions

def load_or_build_extractions(data):
    cf=os.path.join(CACHE,"extractions.json")
    if os.path.exists(cf):
        return json.load(open(cf,encoding="utf-8"))
    all_ex=[]; t0=time.time()
    for si,s in enumerate(data):
        conv_facts=[]
        for sess in session_texts(s):
            conv_facts.append(extract_facts(sess))
        all_ex.append(conv_facts)
        print(f"  extracted conv {si+1}/{len(data)} ({sum(len(x) for x in conv_facts)} facts) t={time.time()-t0:.0f}s",flush=True)
    json.dump(all_ex,open(cf,"w",encoding="utf-8"))
    return all_ex

def load_or_build_embeddings(unique_facts):
    cf=os.path.join(CACHE,"embeddings.json")
    cache=json.load(open(cf,encoding="utf-8")) if os.path.exists(cf) else {}
    missing=[f for f in unique_facts if f not in cache]
    if missing:
        t0=time.time()
        for i,f in enumerate(missing):
            cache[f]=embed(f)
            if (i+1)%200==0: print(f"  embedded {i+1}/{len(missing)} t={time.time()-t0:.0f}s",flush=True)
        json.dump(cache,open(cf,"w",encoding="utf-8"))
    return cache

def clusters_cosine(emb,tau):
    n=emb.shape[0]; sim=emb@emb.T; parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y):
        rx,ry=find(x),find(y);  parent[rx]=ry if rx!=ry else parent[rx]
    iu=np.triu_indices(n,1)
    for p in np.where(sim[iu]>=tau)[0]:
        union(int(iu[0][p]),int(iu[1][p]))
    g={}
    for i in range(n): g.setdefault(find(i),[]).append(i)
    return list(g.values())

def measure(facts,embmap):
    if not facts: return {tau:(0,0,0,0) for tau in TAUS}
    E=np.asarray([embmap[f] for f in facts],dtype=np.float32)
    E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
    tot=len(facts); tottok=sum(tok(f) for f in facts); out={}
    for tau in TAUS:
        cl=clusters_cosine(E,tau); nc=len(cl)
        canon=sum(min(tok(facts[i]) for i in c) for c in cl)
        out[tau]=(tot,nc,1-nc/tot,(tottok-canon)/tottok)
    return out

def build_store(conv_facts,embmap,gated):
    store=[]; store_emb=[]
    for sess in conv_facts:
        for f in sess:
            if f not in embmap: continue
            e=np.asarray(embmap[f],dtype=np.float32); e/=np.linalg.norm(e)+1e-9
            if gated and store_emb:
                if max(float(e@se) for se in store_emb)>=GATE:
                    continue  # NOOP: similarity-gated dedup
            store.append(f); store_emb.append(e)
    return store

def main():
    data=json.load(open(DATA,encoding="utf-8"))
    print("Building LLM-extracted memory store from LoCoMo (qwen3:8b)...",flush=True)
    ex=load_or_build_extractions(data)
    uniq=sorted({f for conv in ex for sess in conv for f in sess})
    print(f"Total extracted facts: {sum(len(s) for c in ex for s in c)} ({len(uniq)} unique strings)",flush=True)
    embmap=load_or_build_embeddings(uniq)

    print("\n"+"="*96)
    print("DEPLOYED-STORE duplication: naive add-all vs Mem0-style gated add (GATE=%.2f)"%GATE)
    print("Store built by LLM extraction on LoCoMo; duplication measured by nomic-embed cosine")
    print("="*96)
    for policy,gated in [("naive",False),("gated",True)]:
        sizes=[]; agg={tau:[0,0,0.0,0.0] for tau in TAUS}
        for conv in ex:
            store=build_store(conv,embmap,gated)
            sizes.append(len(store))
            row=measure(store,embmap)
            for tau in TAUS:
                t,c,m,r=row[tau]; a=agg[tau]; a[0]+=t;a[1]+=c;a[2]+=m;a[3]+=r
        n=len(ex)
        print(f"\n--- policy={policy}  avg store size={sum(sizes)/n:.0f} facts/conv ---")
        print(f"{'tau':>6}{'units':>8}{'canonical':>11}{'MDR':>8}{'RTF':>8}")
        for tau in TAUS:
            t,c,m,r=agg[tau]
            print(f"{tau:>6.2f}{t//n:>8}{c//n:>11}{m/n:>8.3f}{r/n:>8.3f}")
    print("\nInterpretation: MDR under 'naive' = duplication a plain extract-store pipeline accumulates;")
    print("MDR under 'gated' = duplication that SURVIVES a similarity-gated dedup (Mem0-style).")

if __name__=="__main__":
    main()
