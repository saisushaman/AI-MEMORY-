"""
Phase-3f: statistical-power test of the method on MULTI-HOP (LoCoMo cat 1).
Phase-3e showed a directional query_aware>full signal on multihop but only n=43.
Here we use ALL cat-1 questions (~282) at the budgets where the signal appeared
(k in {8,12}), for the three policies, LLM-judged, with a paired McNemar test of
query_aware vs full. Reuses cached answers/judgments; resumable.
"""
import json, os, time, math
import numpy as np
import importlib.util
spec=importlib.util.spec_from_file_location("qd","experiments/qa_dedup_eval.py")
qd=importlib.util.module_from_spec(spec); spec.loader.exec_module(qd)
CACHE=qd.CACHE; KS=[8,12]; POLS=["full","naive_dedup","query_aware"]

def load(name):
    p=os.path.join(CACHE,name); return json.load(open(p,encoding="utf-8")) if os.path.exists(p) else {}
def save(obj,name): json.dump(obj,open(os.path.join(CACHE,name),"w",encoding="utf-8"))

def judge(question,gold,pred):
    prompt=(f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}\n\n"
            "Does the candidate answer convey the same information as the reference answer "
            'to the question? Ignore wording/format. Reply strictly JSON {"correct": true|false}.')
    out=qd._post("/api/chat",{"model":qd.GEN,"messages":[{"role":"user","content":prompt}],
        "stream":False,"think":False,"format":"json","options":{"temperature":0,"seed":0}},300)
    try: return bool(json.loads(out["message"]["content"]).get("correct",False))
    except Exception: return False

def mcnemar_p(b,c):
    # exact two-sided binomial on discordant pairs, H0 p=0.5
    n=b+c
    if n==0: return 1.0
    k=min(b,c)
    from math import comb
    tail=sum(comb(n,i) for i in range(0,k+1))/(2**n)
    return min(1.0,2*tail)

def main():
    data=json.load(open(qd.DATA,encoding="utf-8"))
    ex=json.load(open(os.path.join(CACHE,"extractions.json"),encoding="utf-8"))
    embmap=json.load(open(os.path.join(CACHE,"embeddings.json"),encoding="utf-8"))
    qcache=load("q_embeddings.json"); ans=load("qa_answers_sweep.json"); jud=load("judgments.json")

    correct={(p,k):[] for p in POLS for k in KS}   # per-question 0/1, aligned order
    paired={k:{"b":0,"c":0} for k in KS}           # qa vs full discordant counts
    t0=time.time(); newg=0; newj=0
    for ci,s in enumerate(data):
        facts=qd.conv_store(ex[ci])
        E=np.asarray([embmap[f] for f in facts],dtype=np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
        dd=qd.dedup_store(facts,E,qd.TAU_DEDUP); full_idx=list(range(len(facts)))
        qs=[q for q in s["qa"] if str(q.get("category"))=="1" and q.get("answer")]
        for q in qs:
            question=q["question"]; gold=str(q["answer"])
            if question not in qcache: qcache[question]=qd.embed(question)
            qv=np.asarray(qcache[question],dtype=np.float32); qv/=np.linalg.norm(qv)+1e-9
            perk={}
            for k in KS:
                res={}
                for pol in POLS:
                    if pol=="full": sel=qd.topk_by_relevance(qv,E,full_idx,k)
                    elif pol=="naive_dedup": sel=qd.topk_by_relevance(qv,E,dd,k)
                    else: sel=qd.mmr(qv,E,full_idx,k,qd.LAMBDA)
                    akey=f"{ci}|{question}|{pol}|{k}"
                    if akey not in ans:
                        ctx="\n".join(f"- {facts[i]}" for i in sel)
                        ans[akey]=qd.answer(ctx,question); newg+=1
                    jkey=akey
                    if jkey not in jud:
                        jud[jkey]=judge(question,gold,ans[akey]); newj+=1
                    ok=1 if jud[jkey] else 0
                    correct[(pol,k)].append(ok); res[pol]=ok
                    if (newg+newj)%100==0 and (newg+newj)>0:
                        save(ans,"qa_answers_sweep.json"); save(jud,"judgments.json"); save(qcache,"q_embeddings.json")
                        print(f"  gen={newg} judge={newj} t={time.time()-t0:.0f}s",flush=True)
                # paired qa vs full
                if res["query_aware"]==1 and res["full"]==0: paired[k]["b"]+=1
                elif res["full"]==1 and res["query_aware"]==0: paired[k]["c"]+=1
        print(f"conv {ci+1}/10 t={time.time()-t0:.0f}s",flush=True)
    save(ans,"qa_answers_sweep.json"); save(jud,"judgments.json"); save(qcache,"q_embeddings.json")

    n=len(correct[("full",KS[0])])
    print("\n"+"="*66)
    print(f"MULTI-HOP (LoCoMo cat 1) statistical test, n={n} questions, qwen3:8b judge")
    print("="*66)
    print(f"{'policy':<13}"+"".join(f"k={k:>2}     " for k in KS))
    for pol in POLS:
        row=f"{pol:<13}"
        for k in KS:
            acc=sum(correct[(pol,k)])/max(1,len(correct[(pol,k)]))
            row+=f"{acc:>6.3f}    "
        print(row)
    print("\nPaired McNemar test  query_aware vs full  (b=qa-only-correct, c=full-only-correct):")
    for k in KS:
        b,c=paired[k]["b"],paired[k]["c"]; p=mcnemar_p(b,c)
        print(f"  k={k}: b={b} c={c}  p={p:.3f}  {'SIGNIFICANT' if p<0.05 else 'not significant'} (favouring {'query_aware' if b>c else 'full'})")

if __name__=="__main__":
    main()
